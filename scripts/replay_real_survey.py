#!/usr/bin/env python3
"""Replay the four real surveyed areas through the revised gate.

Reconstructs each area from its published survey aggregates -- year histogram,
sequence structure, median baseline -- and re-runs the gate. Approximate by
construction: exact per-frame data is not reproduced, only the distributions
the gate actually reads.

    PYTHONPATH=src python3 scripts/replay_real_survey.py

Purpose is to check the revised thresholds against real numbers rather than
synthetic ones, and to keep a record of what changed and why.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachable.geo import BBox  # noqa: E402
from reachable.models import ImageRecord  # noqa: E402
from reachable.survey import format_report, survey_images  # noqa: E402

DEG_PER_M_LAT = 1.0 / 111_195.0
NOW = datetime.now(tz=timezone.utc)

# Surveyed 2026-08-30. bbox, year histogram, sequence count, count of
# sequences with >=5 frames, median inter-frame baseline.
AREAS = {
    "campus_core": {
        "bbox": "-86.9302,40.4167,-86.9122,40.4307",
        "years": {2015: 60, 2016: 12, 2020: 680, 2021: 1225, 2025: 27},
        "n_seq": 71, "n_long": 19, "baseline_m": 8.54,
    },
    "chauncey_village": {
        "bbox": "-86.915,40.4175,-86.897,40.4315",
        "years": {2015: 751, 2016: 21, 2018: 434, 2020: 828,
                  2021: 1962, 2024: 3, 2025: 68},
        "n_seq": 132, "n_long": 62, "baseline_m": 8.63,
    },
    "wl_residential_north": {
        "bbox": "-86.929,40.438,-86.911,40.452",
        "years": {2015: 253, 2016: 102, 2018: 541, 2021: 224},
        "n_seq": 18, "n_long": 16, "baseline_m": 4.97,
    },
    "downtown_lafayette": {
        "bbox": "-86.8843,40.4097,-86.8663,40.4237",
        "years": {2020: 127, 2021: 663},
        "n_seq": 6, "n_long": 6, "baseline_m": 10.86,
    },
}


def build(spec) -> list[ImageRecord]:
    """Synthesise frames matching the area's year and sequence distribution.

    Frames are allocated newest-year-first into long sequences, so the recent
    layer lands inside multi-view runs -- the most favourable arrangement for
    the area. If it still fails, it fails on the data, not on the layout.
    """
    bbox = BBox.from_param(spec["bbox"])
    n_long, n_seq = spec["n_long"], spec["n_seq"]
    total = sum(spec["years"].values())
    long_frames = total - (n_seq - n_long)  # orphans take one frame each
    per_long = max(5, long_frames // max(n_long, 1))

    records, idx = [], 0
    for year in sorted(spec["years"], reverse=True):
        remaining = spec["years"][year]
        # Mid-year timestamp; good enough for age bucketing.
        ts = datetime(year, 6, 15, tzinfo=timezone.utc)
        if ts > NOW:
            ts = NOW - timedelta(days=30)
        while remaining > 0:
            seq_idx = idx // per_long
            in_long = seq_idx < n_long
            take = min(remaining, per_long if in_long else 1)
            sid = f"seq{seq_idx}" if in_long else f"solo{idx}"
            for f in range(take):
                lat = bbox.south + 0.002 + (
                    (idx % 400) * spec["baseline_m"] * DEG_PER_M_LAT
                )
                records.append(ImageRecord(
                    id=f"i{idx}", sequence_id=sid,
                    captured_at_ms=int(ts.timestamp() * 1000) + idx * 1000,
                    lon=bbox.west + 0.002 + (seq_idx % 40) * 0.0002, lat=lat,
                    compass_angle=90.0, altitude=190.0,
                    computed_lon=bbox.west + 0.002 + (seq_idx % 40) * 0.0002,
                    computed_lat=lat,
                    computed_compass_angle=90.0, computed_altitude=190.0,
                    camera_type="perspective", camera_parameters=[0.85, 0.0, 0.0],
                    width=2048, height=1536, is_pano=False, exif_orientation=1,
                    thumb_1024_url="https://x.invalid/a.jpg",
                    thumb_2048_url="https://x.invalid/b.jpg",
                ))
                idx += 1
            remaining -= take
    return records


def main() -> int:
    results = []
    for name, spec in AREAS.items():
        results.append(
            survey_images(name, BBox.from_param(spec["bbox"]), build(spec))
        )

    print(format_report(results))

    print("\nRECENCY SUMMARY (the decisive number)\n")
    print(f"{'area':<24} {'newest':>9} {'recent':>8} {'deep':>7}  verdict")
    print("-" * 66)
    for r in results:
        print(f"{r.area_name:<24} {str(r.newest_age_years) + 'y':>9} "
              f"{r.recent_fraction:>7.1%} {r.deep_frames:>7}  {r.verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())