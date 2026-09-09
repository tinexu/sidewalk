"""Stage 0 ingest: fetch imagery and cache it, partitioned by sequence.

Design constraints that shaped this:

1. **Resumability.** A study area is thousands of downloads over tens of
   minutes. Anything that cannot resume after a dropped connection will cost a
   day at some point. Every image is checked against storage before fetching,
   and the manifest is written incrementally.

2. **Thumb URLs expire.** They are pre-signed CDN links, so metadata and bytes
   are fetched in the same pass. Caching URLs to download "later" produces a
   pile of 403s.

3. **Sequence partitioning.** Stage 3 reads whole sequences to build baselines.
   Laying bytes out by sequence gives it contiguous reads and lets AWS Batch
   shard by sequence with no cross-shard coordination.

4. **Verify before commit.** A truncated download that lands in the cache is
   worse than no download, because every later run treats it as a hit. Bytes
   are decoded before they are stored.
"""

from __future__ import annotations

import io
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from PIL import Image as PILImage

from .geo import BBox
from .mapillary import MapillaryClient, MapillaryError
from .models import ImageRecord, summarize_sequences, write_jsonl
from .storage import Storage, image_key, manifest_key, metadata_key, sha256_hex

log = logging.getLogger(__name__)


@dataclass
class IngestStats:
    started_at: str = ""
    finished_at: str = ""
    metadata_records: int = 0
    already_cached: int = 0
    downloaded: int = 0
    failed: int = 0
    corrupt: int = 0
    no_url: int = 0
    bytes_downloaded: int = 0
    sequences: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def verify_image_bytes(data: bytes) -> tuple[bool, int, int]:
    """Decode-verify a download. Returns (ok, width, height).

    PIL's verify() catches truncation and header corruption without decoding
    the full raster, which is fast enough to run on every image. A load() is
    then needed to read dimensions, since verify() leaves the file unusable.
    """
    try:
        PILImage.open(io.BytesIO(data)).verify()
        im = PILImage.open(io.BytesIO(data))
        return True, im.width, im.height
    except Exception as exc:  # noqa: BLE001 - any decode failure is a reject
        log.debug("image verification failed: %s", exc)
        return False, 0, 0


class Ingestor:
    def __init__(
        self,
        client: MapillaryClient,
        storage: Storage,
        workers: int = 8,
        skip_panos: bool = False,
        min_width: int = 1024,
    ) -> None:
        self.client = client
        self.storage = storage
        self.workers = workers
        self.skip_panos = skip_panos
        self.min_width = min_width

    # -- metadata ----------------------------------------------------------

    def fetch_metadata(
        self, bbox: BBox, tile_size: float = 0.009
    ) -> list[ImageRecord]:
        """Pull all image metadata for the area. No bytes yet."""
        t0 = time.monotonic()
        records = list(self.client.images_in_bbox(bbox, tile_size=tile_size))
        log.info(
            "fetched %d metadata records in %.1fs (%d API requests)",
            len(records),
            time.monotonic() - t0,
            self.client.request_count,
        )
        return records

    def filter_records(self, records: list[ImageRecord]) -> list[ImageRecord]:
        """Drop frames that cannot contribute to Stage 3.

        Deliberately conservative: this is not the quality triage of Stage 1,
        only the structural filter. Anything dropped here could never be used
        no matter how good the pixels are.
        """
        keep: list[ImageRecord] = []
        for r in records:
            if not r.sequence_id:
                continue
            if r.best_lon is None or r.best_lat is None:
                continue
            if r.width is not None and r.width < self.min_width:
                continue
            if self.skip_panos and r.is_pano:
                continue
            if not r.download_url():
                continue
            keep.append(r)
        log.info("kept %d of %d records after structural filter",
                 len(keep), len(records))
        return keep

    # -- bytes -------------------------------------------------------------

    def _ingest_one(self, rec: ImageRecord) -> tuple[str, ImageRecord, str | None]:
        """Fetch and store one image. Returns (outcome, record, error)."""
        key = image_key(rec.sequence_id, rec.id)

        if self.storage.exists(key):
            rec.stored_key = key
            return ("cached", rec, None)

        url = rec.download_url()
        if not url:
            return ("no_url", rec, "no thumb url in metadata")

        try:
            data = self.client.fetch_image_bytes(url)
        except MapillaryError as exc:
            return ("failed", rec, str(exc))

        ok, w, h = verify_image_bytes(data)
        if not ok:
            return ("corrupt", rec, "failed decode verification")

        self.storage.put_bytes(key, data, "image/jpeg")
        rec.stored_key = key
        rec.stored_bytes = len(data)
        rec.stored_sha256 = sha256_hex(data)
        # Trust the decoded dimensions over the API's, which occasionally
        # describe the original rather than the thumb.
        rec.width, rec.height = w, h
        return ("downloaded", rec, None)

    def ingest_images(
        self,
        records: list[ImageRecord],
        progress_every: int = 100,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> tuple[list[ImageRecord], IngestStats]:
        stats = IngestStats(
            started_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        )
        stats.metadata_records = len(records)
        done: list[ImageRecord] = []

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._ingest_one, r): r for r in records}
            for i, fut in enumerate(as_completed(futures), 1):
                outcome, rec, err = fut.result()

                if outcome == "cached":
                    stats.already_cached += 1
                    done.append(rec)
                elif outcome == "downloaded":
                    stats.downloaded += 1
                    stats.bytes_downloaded += rec.stored_bytes or 0
                    done.append(rec)
                elif outcome == "corrupt":
                    stats.corrupt += 1
                elif outcome == "no_url":
                    stats.no_url += 1
                else:
                    stats.failed += 1

                if err and len(stats.failures) < 50:
                    stats.failures.append({"image_id": rec.id, "error": err[:200]})

                if progress_every and i % progress_every == 0:
                    log.info(
                        "%d/%d  downloaded=%d cached=%d failed=%d",
                        i, len(records), stats.downloaded,
                        stats.already_cached, stats.failed,
                    )
                    if on_progress:
                        on_progress(i, len(records))

        stats.finished_at = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        return done, stats

    # -- metadata persistence ---------------------------------------------

    def write_sequence_metadata(self, records: list[ImageRecord]) -> int:
        """Write one JSONL per sequence, frames ordered by capture time.

        Ordering here means Stage 3 can stream a sequence in capture order
        without re-sorting, and the ordering rule lives in exactly one place.
        """
        sequences = summarize_sequences(records)
        by_id = {r.id: r for r in records}

        for sid, seq in sequences.items():
            ordered = [by_id[i] for i in seq.image_ids if i in by_id]
            payload = "\n".join(
                json.dumps(r.to_dict(), separators=(",", ":")) for r in ordered
            )
            self.storage.put_text(
                metadata_key(sid), payload, "application/x-ndjson"
            )
        return len(sequences)

    def write_manifest(
        self,
        area: str,
        bbox: BBox,
        records: list[ImageRecord],
        stats: IngestStats,
    ) -> dict[str, Any]:
        """Manifest is the contract between Stage 0 and everything downstream.

        It also carries the provenance and licensing record that the report's
        responsible-operation section needs.
        """
        sequences = summarize_sequences(records)
        stats.sequences = len(sequences)

        manifest = {
            "schema_version": 1,
            "area": area,
            "bbox": bbox.to_param(),
            "area_km2": round(bbox.area_km2(), 3),
            "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "source": {
                "provider": "Mapillary",
                "api": "graph.mapillary.com/images",
                "license": "CC BY-SA 4.0",
                "attribution_required": True,
                "attribution_note": (
                    "Display the Mapillary logo with a link to mapillary.com "
                    "wherever data extracted via the API is integrated."
                ),
                "privacy_note": (
                    "Mapillary blurs faces and licence plates before "
                    "publication. No additional PII is stored by this pipeline."
                ),
            },
            "counts": {
                "images": len(records),
                "sequences": len(sequences),
                "bytes": stats.bytes_downloaded,
            },
            "sequences": {
                sid: {"n_images": s.length, "metadata_key": metadata_key(sid)}
                for sid, s in sorted(sequences.items())
            },
            "ingest_stats": stats.to_dict(),
            "api_stats": self.client.stats(),
        }
        self.storage.put_json(manifest_key(area), manifest)
        return manifest

    # -- orchestration -----------------------------------------------------

    def run(
        self,
        area: str,
        bbox: BBox,
        tile_size: float = 0.009,
        limit: int | None = None,
        local_metadata_path: str | None = None,
    ) -> dict[str, Any]:
        log.info("=== Stage 0 ingest: %s ===", area)
        log.info("bbox %s (%.2f km^2)", bbox.to_param(), bbox.area_km2())

        raw = self.fetch_metadata(bbox, tile_size=tile_size)
        records = self.filter_records(raw)

        if limit is not None and len(records) > limit:
            log.info("limiting to %d records (--limit)", limit)
            records = records[:limit]

        stored, stats = self.ingest_images(records)
        n_seq = self.write_sequence_metadata(stored)
        log.info("wrote metadata for %d sequences", n_seq)

        manifest = self.write_manifest(area, bbox, stored, stats)

        if local_metadata_path:
            write_jsonl(stored, local_metadata_path)
            log.info("wrote local metadata copy to %s", local_metadata_path)

        log.info(
            "done: %d downloaded, %d already cached, %d failed, %d corrupt, %.2f GB",
            stats.downloaded, stats.already_cached, stats.failed,
            stats.corrupt, stats.bytes_downloaded / 1e9,
        )
        return manifest