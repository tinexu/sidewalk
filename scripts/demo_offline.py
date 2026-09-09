#!/usr/bin/env python3
"""Offline end-to-end demo of Stage 0. No network, no API token.

Runs the real survey and ingest code paths against a synthetic Mapillary API
so the whole pipeline can be exercised in CI, in a sandbox, or by a judge
reproducing the build without credentials.

    PYTHONPATH=src python3 scripts/demo_offline.py

Two synthetic areas are generated to show the gate discriminating: one with
dense sequences at usable frame spacing, one with the sparse, widely-spaced
coverage typical of a highway corridor.
"""

from __future__ import annotations

import io
import json
import logging
import sys
import tempfile
from pathlib import Path

from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachable.geo import BBox  # noqa: E402
from reachable.ingest import Ingestor  # noqa: E402
from reachable.mapillary import MapillaryClient  # noqa: E402
from reachable.models import ImageRecord  # noqa: E402
from reachable.storage import LocalStorage, manifest_key  # noqa: E402
from reachable.survey import format_report, survey_images  # noqa: E402

DEG_PER_M_LAT = 1.0 / 111_195.0
BBOX = BBox(-86.9302, 40.4167, -86.9122, 40.4307)


def synth_api_image(idx, seq, lon, lat, ts_ms):
    return {
        "id": f"img{idx}",
        "sequence": seq,
        "captured_at": ts_ms,
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


def synth_dataset(n_seq, frames, spacing_m, start_idx=0, base_ts=1_750_000_000_000):
    out, idx = [], start_idx
    for s in range(n_seq):
        lon = -86.9280 + (s % 12) * 0.0012
        lat0 = 40.4180 + (s // 12) * 0.0018
        for f in range(frames):
            out.append(
                synth_api_image(
                    idx, f"seq{s}", lon,
                    lat0 + f * spacing_m * DEG_PER_M_LAT,
                    base_ts + idx * 2000,
                )
            )
            idx += 1
    return out


class SynthAPI:
    def __init__(self, images, page_size=400):
        self.images, self.page_size = images, page_size
        self._bbox = None
        self.request_count = 0

    def get(self, url, params=None):
        self.request_count += 1
        if params and "bbox" in params:
            self._bbox = tuple(float(x) for x in params["bbox"].split(","))
            offset = 0
        else:
            offset = int(url.rsplit("offset=", 1)[1])
        w, s, e, n = self._bbox
        matched = [
            im for im in self.images
            if w <= im["geometry"]["coordinates"][0] <= e
            and s <= im["geometry"]["coordinates"][1] <= n
        ]
        page = matched[offset : offset + self.page_size]
        out = {"data": page}
        if offset + self.page_size < len(matched):
            out["paging"] = {
                "next": f"https://synth.invalid/images?offset={offset + self.page_size}"
            }
        return out

    def fetch_bytes(self, url):
        buf = io.BytesIO()
        PILImage.new("RGB", (2048, 1536), (118, 126, 108)).save(
            buf, format="JPEG", quality=60
        )
        return buf.getvalue()


def wire(api):
    client = MapillaryClient(token="demo", rate=0.0)
    client._get = lambda url, params=None: api.get(url, params)
    client.fetch_image_bytes = api.fetch_bytes
    return client


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    print("\n" + "#" * 72)
    print("# PART 1 - COVERAGE SURVEY (metadata only, the day-1 gate)")
    print("#" * 72)

    scenarios = {
        # Dense residential capture: many sequences, 3 m frame spacing.
        "wl_residential_north": synth_dataset(48, 22, spacing_m=3.0),
        # Highway-style: plenty of frames, but 35 m apart. Looks like
        # abundance on an image count and is useless for triangulation.
        "us231_corridor": synth_dataset(30, 30, spacing_m=35.0, start_idx=100_000),
    }

    results = []
    for name, data in scenarios.items():
        api = SynthAPI(data)
        client = wire(api)
        images = list(client.images_in_bbox(BBOX, progress=False))
        results.append(survey_images(name, BBOX, images))
        print(f"  surveyed {name}: {len(images)} images, "
              f"{api.request_count} API requests")

    print()
    print(format_report(results))

    print("\n" + "#" * 72)
    print("# PART 2 - INGEST (metadata + bytes, cached by sequence)")
    print("#" * 72 + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        api = SynthAPI(scenarios["wl_residential_north"])
        client = wire(api)
        storage = LocalStorage(Path(tmp) / "cache")
        ingestor = Ingestor(client, storage, workers=8)

        manifest = ingestor.run("wl_residential_north", BBOX, limit=120)

        print("\n--- cache layout (first 6 keys) ---")
        for k in storage.list_keys("imagery/")[:6]:
            print(f"  {k}")
        print(f"  ... {len(storage.list_keys('imagery/'))} image files total")
        print("\n--- metadata keys (first 3) ---")
        for k in storage.list_keys("metadata/")[:3]:
            print(f"  {k}")

        print("\n--- manifest excerpt ---")
        m = storage.get_json(manifest_key("wl_residential_north"))
        print(json.dumps(
            {"area": m["area"], "bbox": m["bbox"], "area_km2": m["area_km2"],
             "counts": m["counts"], "source": m["source"]},
            indent=2,
        ))

        print("\n--- resumability check: immediate re-run ---")
        again = ingestor.run("wl_residential_north", BBOX, limit=120)
        s = again["ingest_stats"]
        print(f"  downloaded={s['downloaded']}  cached={s['already_cached']}  "
              f"failed={s['failed']}")
        assert s["downloaded"] == 0, "re-run should download nothing"
        print("  OK: second run downloaded nothing, all served from cache.")

    print("\n" + "=" * 72)
    print("Stage 0 offline demo complete.")
    print("=" * 72 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())