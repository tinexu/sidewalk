"""Ingest tests against a mock Graph API.

This sandbox cannot reach graph.mapillary.com, and neither can CI without a
token, so the API is mocked at the transport boundary: the mock implements
bbox filtering, pagination and thumb-URL serving, and everything above it is
the real code path. That exercises tiling, dedup, resumability, corruption
rejection and manifest generation without a network.

What this cannot verify is the API's actual field names, response shape and
error codes. Those need a live smoke test on day 1 -- see README.
"""

import io
import json
from pathlib import Path

import pytest
from PIL import Image as PILImage

from reachable.geo import BBox
from reachable.ingest import Ingestor, verify_image_bytes
from reachable.mapillary import MapillaryClient, MapillaryError
from reachable.storage import LocalStorage, image_key, manifest_key
from reachable.models import IMAGE_FIELDS

DEG_PER_M_LAT = 1.0 / 111_195.0


def jpeg_bytes(w: int = 2048, h: int = 1536, color=(120, 130, 110)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (w, h), color).save(buf, format="JPEG", quality=70)
    return buf.getvalue()


class MockGraphAPI:
    """Stand-in for graph.mapillary.com/images."""

    def __init__(self, images: list[dict], page_size: int = 500):
        self.images = images
        self.page_size = page_size
        self.requests: list[dict] = []
        self.byte_fetches = 0
        self.fail_ids: set[str] = set()
        self.corrupt_ids: set[str] = set()
        self._active_bbox: tuple[float, float, float, float] | None = None

    def _match(self) -> list[dict]:
        w, s, e, n = self._active_bbox
        return [
            im for im in self.images
            if w <= im["geometry"]["coordinates"][0] <= e
            and s <= im["geometry"]["coordinates"][1] <= n
        ]

    def get(self, url: str, params: dict | None) -> dict:
        self.requests.append({"url": url, "params": params})

        if params and "bbox" in params:
            w, s, e, n = (float(x) for x in params["bbox"].split(","))
            if (e - w) >= 0.01 or (n - s) >= 0.01:
                raise MapillaryError("Bad request (400): bbox too large")
            # Hold the bbox for the duration of this tile's pagination. Reading
            # it back off the previous request breaks as soon as two paged
            # requests arrive consecutively.
            self._active_bbox = (w, s, e, n)
            offset = 0
        else:
            # Follow-on page: offset is encoded in the synthetic `next` URL.
            offset = int(url.rsplit("offset=", 1)[1])
            if self._active_bbox is None:
                raise AssertionError("paged request with no preceding bbox query")

        matched = self._match()
        page = matched[offset : offset + self.page_size]
        out: dict = {"data": page}
        if offset + self.page_size < len(matched):
            nxt = offset + self.page_size
            out["paging"] = {"next": f"https://mock.invalid/images?offset={nxt}"}
        return out

    def fetch_bytes(self, url: str) -> bytes:
        self.byte_fetches += 1
        image_id = url.rsplit("/", 1)[-1].split(".")[0]
        if image_id in self.fail_ids:
            raise MapillaryError("HTTP 500 fetching image bytes")
        if image_id in self.corrupt_ids:
            return b"\xff\xd8\xff\xe0 truncated garbage not a real jpeg"
        return jpeg_bytes()


def make_api_image(idx: int, seq: str, lon: float, lat: float, **over) -> dict:
    d = {
        "id": f"img{idx}",
        "sequence": seq,
        "captured_at": 1_750_000_000_000 + idx * 2000,
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "computed_geometry": {"type": "Point", "coordinates": [lon, lat]},
        "compass_angle": 88.0,
        "computed_compass_angle": 90.0,
        "altitude": 190.0,
        "computed_altitude": 190.5,
        "camera_type": "perspective",
        "camera_parameters": [0.85, 0.0, 0.0],
        "width": 2048,
        "height": 1536,
        "is_pano": False,
        "exif_orientation": 1,
        "thumb_1024_url": f"https://cdn.invalid/1024/img{idx}.jpg",
        "thumb_2048_url": f"https://cdn.invalid/2048/img{idx}.jpg",
    }
    d.update(over)
    return d


def build_dataset(n_seq: int = 6, frames: int = 25, spacing_m: float = 3.0):
    out, idx = [], 0
    for s in range(n_seq):
        lon = -86.9250 + s * 0.0015
        for f in range(frames):
            out.append(
                make_api_image(idx, f"seq{s}", lon,
                               40.4200 + f * spacing_m * DEG_PER_M_LAT)
            )
            idx += 1
    return out


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Client + storage + ingestor wired to a mock API."""
    api = MockGraphAPI(build_dataset())
    client = MapillaryClient(token="test-token", rate=0.0, max_retries=2)
    monkeypatch.setattr(client, "_get", lambda url, params=None: api.get(url, params))
    monkeypatch.setattr(client, "fetch_image_bytes", api.fetch_bytes)
    storage = LocalStorage(tmp_path / "cache")
    ingestor = Ingestor(client=client, storage=storage, workers=4)
    return api, client, storage, ingestor, tmp_path


BBOX = BBox(-86.9260, 40.4190, -86.9160, 40.4215)


class TestVerification:
    def test_valid_jpeg_accepted(self):
        ok, w, h = verify_image_bytes(jpeg_bytes(800, 600))
        assert ok and (w, h) == (800, 600)

    def test_truncated_jpeg_rejected(self):
        ok, _, _ = verify_image_bytes(jpeg_bytes()[:200])
        assert not ok

    def test_non_image_rejected(self):
        assert not verify_image_bytes(b"<html>404 not found</html>")[0]

    def test_empty_rejected(self):
        assert not verify_image_bytes(b"")[0]


class TestMetadataFetch:
    def test_all_tiles_are_compliant(self, wired):
        api, _, _, ingestor, _ = wired
        ingestor.fetch_metadata(BBOX)
        for req in api.requests:
            if req["params"] and "bbox" in req["params"]:
                w, s, e, n = (float(x) for x in req["params"]["bbox"].split(","))
                assert (e - w) < 0.01 and (n - s) < 0.01

    def test_requests_the_declared_fields(self, wired):
        api, _, _, ingestor, _ = wired
        ingestor.fetch_metadata(BBOX)
        fields = api.requests[0]["params"]["fields"].split(",")
        assert set(fields) == set(IMAGE_FIELDS)

    def test_deduplicates_across_tile_boundaries(self, wired):
        _, _, _, ingestor, _ = wired
        recs = ingestor.fetch_metadata(BBOX)
        ids = [r.id for r in recs]
        assert len(ids) == len(set(ids))

    def test_pagination_is_followed(self, monkeypatch, tmp_path):
        api = MockGraphAPI(build_dataset(n_seq=4, frames=60), page_size=25)
        client = MapillaryClient(token="t", rate=0.0)
        monkeypatch.setattr(client, "_get",
                            lambda url, params=None: api.get(url, params))
        ing = Ingestor(client, LocalStorage(tmp_path / "c"))
        recs = ing.fetch_metadata(BBox(-86.9260, 40.4190, -86.9160, 40.4215))
        assert len(recs) > 25  # more than one page was consumed
        assert any(r["params"] is None for r in api.requests)

    def test_prefers_computed_geometry(self, wired):
        _, _, _, ingestor, _ = wired
        recs = ingestor.fetch_metadata(BBOX)
        r = recs[0]
        assert r.best_heading == r.computed_compass_angle == 90.0
        assert r.best_heading != r.compass_angle

    def test_oversized_bbox_rejected_before_request(self, wired):
        _, client, _, _, _ = wired
        with pytest.raises(MapillaryError, match="0.01"):
            list(client.images_in_tile(BBox(0, 0, 0.05, 0.05)))


class TestIngestRun:
    def test_end_to_end(self, wired):
        _, _, storage, ingestor, _ = wired
        manifest = ingestor.run("test_area", BBOX)

        assert manifest["counts"]["images"] > 0
        assert manifest["counts"]["sequences"] > 0
        assert manifest["ingest_stats"]["failed"] == 0
        assert manifest["ingest_stats"]["corrupt"] == 0
        assert storage.exists(manifest_key("test_area"))

    def test_bytes_land_under_sequence_partitions(self, wired):
        _, _, storage, ingestor, _ = wired
        ingestor.run("test_area", BBOX)
        keys = storage.list_keys("imagery/")
        assert keys
        assert all(k.startswith("imagery/seq=") for k in keys)
        assert all(k.endswith(".jpg") for k in keys)

    def test_sequence_metadata_is_ordered_by_capture_time(self, wired):
        _, _, storage, ingestor, _ = wired
        ingestor.run("test_area", BBOX)
        for key in storage.list_keys("metadata/"):
            rows = [json.loads(l) for l in
                    storage.get_bytes(key).decode().splitlines() if l]
            times = [r["captured_at_ms"] for r in rows]
            assert times == sorted(times)

    def test_second_run_is_fully_cached(self, wired):
        _, _, _, ingestor, _ = wired
        first = ingestor.run("test_area", BBOX)
        second = ingestor.run("test_area", BBOX)
        assert first["ingest_stats"]["downloaded"] > 0
        assert second["ingest_stats"]["downloaded"] == 0
        assert second["ingest_stats"]["already_cached"] == \
            first["ingest_stats"]["downloaded"]

    def test_resumes_after_partial_failure(self, wired):
        api, _, storage, ingestor, _ = wired
        recs = ingestor.filter_records(ingestor.fetch_metadata(BBOX))
        api.fail_ids = {r.id for r in recs[:10]}

        _, stats1 = ingestor.ingest_images(recs)
        assert stats1.failed == 10
        assert stats1.downloaded == len(recs) - 10

        api.fail_ids = set()  # transient outage clears
        _, stats2 = ingestor.ingest_images(recs)
        assert stats2.failed == 0
        assert stats2.downloaded == 10
        assert stats2.already_cached == len(recs) - 10

    def test_corrupt_downloads_are_not_cached(self, wired):
        api, _, storage, ingestor, _ = wired
        recs = ingestor.filter_records(ingestor.fetch_metadata(BBOX))
        bad = recs[0]
        api.corrupt_ids = {bad.id}

        _, stats = ingestor.ingest_images(recs)
        assert stats.corrupt == 1
        # Critical: a corrupt file must not become a cache hit next run.
        assert not storage.exists(image_key(bad.sequence_id, bad.id))

    def test_limit_caps_downloads(self, wired):
        api, _, _, ingestor, _ = wired
        manifest = ingestor.run("test_area", BBOX, limit=20)
        assert manifest["counts"]["images"] == 20
        assert api.byte_fetches == 20

    def test_manifest_carries_licensing_provenance(self, wired):
        _, _, storage, ingestor, _ = wired
        ingestor.run("test_area", BBOX)
        m = storage.get_json(manifest_key("test_area"))
        assert m["source"]["license"] == "CC BY-SA 4.0"
        assert m["source"]["attribution_required"] is True
        assert "mapillary.com" in m["source"]["attribution_note"]
        assert "blurs faces" in m["source"]["privacy_note"]

    def test_manifest_indexes_every_sequence(self, wired):
        _, _, storage, ingestor, _ = wired
        m = ingestor.run("test_area", BBOX)
        for sid, entry in m["sequences"].items():
            assert entry["n_images"] > 0
            assert storage.exists(entry["metadata_key"])


class TestFiltering:
    def test_drops_records_without_sequence(self, monkeypatch, tmp_path):
        data = build_dataset(n_seq=2, frames=10)
        data.append(make_api_image(999, None, -86.9200, 40.4200, sequence=None))
        api = MockGraphAPI(data)
        client = MapillaryClient(token="t", rate=0.0)
        monkeypatch.setattr(client, "_get",
                            lambda url, params=None: api.get(url, params))
        ing = Ingestor(client, LocalStorage(tmp_path / "c"))
        recs = ing.filter_records(ing.fetch_metadata(BBOX))
        assert all(r.sequence_id for r in recs)

    def test_skip_panos_flag(self, monkeypatch, tmp_path):
        data = build_dataset(n_seq=2, frames=10)
        for d in data[:5]:
            d["is_pano"] = True
        api = MockGraphAPI(data)
        client = MapillaryClient(token="t", rate=0.0)
        monkeypatch.setattr(client, "_get",
                            lambda url, params=None: api.get(url, params))
        ing = Ingestor(client, LocalStorage(tmp_path / "c"), skip_panos=True)
        recs = ing.filter_records(ing.fetch_metadata(BBOX))
        assert not any(r.is_pano for r in recs)


class TestStorageSafety:
    def test_key_traversal_blocked(self, tmp_path):
        s = LocalStorage(tmp_path / "cache")
        with pytest.raises(ValueError, match="escapes"):
            s.put_bytes("../../etc/passwd", b"x", "text/plain")

    def test_sibling_prefix_not_mistaken_for_child(self, tmp_path):
        """A sibling sharing a name prefix must not pass the containment check.

        str.startswith would accept /data/cache-evil against a /data/cache
        root. is_relative_to compares path components and does not.
        """
        (tmp_path / "cache-evil").mkdir()
        s = LocalStorage(tmp_path / "cache")
        with pytest.raises(ValueError, match="escapes"):
            s.put_bytes("../cache-evil/x.jpg", b"x", "image/jpeg")

    def test_works_through_a_symlinked_root(self, tmp_path):
        """Regression: macOS /var is a symlink to /private/var.

        A tempfile root arrives as /var/folders/... while .resolve() on any key
        beneath it yields /private/var/... . Storing the root unresolved while
        resolving keys made relative_to() raise on every list_keys call, so the
        demo blew up on macOS and passed everywhere else. The root is now
        resolved once at construction.
        """
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)

        s = LocalStorage(link / "cache")
        s.put_bytes("imagery/seq=a/1.jpg", jpeg_bytes(64, 64), "image/jpeg")
        s.put_bytes("imagery/seq=b/2.jpg", jpeg_bytes(64, 64), "image/jpeg")

        assert s.list_keys("imagery/") == [
            "imagery/seq=a/1.jpg",
            "imagery/seq=b/2.jpg",
        ]
        assert s.exists("imagery/seq=a/1.jpg")
        assert s.get_bytes("imagery/seq=a/1.jpg")

    def test_full_ingest_through_a_symlinked_root(self, monkeypatch, tmp_path):
        """The failure surfaced in the demo, so cover the whole run, not just
        the storage unit."""
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)

        api = MockGraphAPI(build_dataset(n_seq=2, frames=10))
        client = MapillaryClient(token="t", rate=0.0)
        monkeypatch.setattr(client, "_get",
                            lambda url, params=None: api.get(url, params))
        monkeypatch.setattr(client, "fetch_image_bytes", api.fetch_bytes)

        storage = LocalStorage(link / "cache")
        ing = Ingestor(client, storage, workers=2)
        manifest = ing.run("symlinked", BBOX)

        assert manifest["counts"]["images"] > 0
        assert storage.list_keys("imagery/")
        assert storage.list_keys("metadata/")

    def test_no_partial_files_left_behind(self, tmp_path):
        s = LocalStorage(tmp_path / "cache")
        s.put_bytes("imagery/seq=a/1.jpg", jpeg_bytes(), "image/jpeg")
        assert not list(Path(tmp_path / "cache").rglob("*.tmp"))