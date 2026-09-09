"""Coverage survey: the day-1 gate.

Before writing a line of geometry code, answer one question: does enough usable
imagery exist over the candidate study areas to build this at all? If not, the
plan changes to own-capture and it changes today, not in week four.

The survey pulls metadata only -- no image bytes -- so it is cheap enough to
run over several candidate areas and pick the best one.

The decisive metric is not image count. It is **median inter-frame spacing
within a sequence**. Stage 3 recovers metric geometry by triangulating between
consecutive frames, and that needs a baseline in a usable band:

  - too short (< ~1 m): triangulation is ill-conditioned, depth error explodes
  - too long (> ~10 m): viewpoint change defeats descriptor matching, and the
    same curb may not appear in both frames at all

A study area with 5,000 images at 25 m spacing is worse than one with 800
images at 3 m spacing. Counting photos alone would pick the wrong area.
"""

from __future__ import annotations

import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from .geo import BBox, count_tiles, haversine_m
from .models import ImageRecord, summarize_sequences

log = logging.getLogger(__name__)

# Baseline band that Stage 3 can actually work with.
MIN_USEFUL_BASELINE_M = 1.0
MAX_USEFUL_BASELINE_M = 10.0

# Gate thresholds.
GATE_MIN_USABLE_IMAGES = 500
GATE_MIN_GOOD_BASELINE_FRAC = 0.35

# Frames living inside sequences of >= MIN_SEQUENCE_FRAMES, summed.
#
# Replaces a count of qualifying sequences. Counting sequences failed
# campus_core at 19 against a threshold of 20 while it sat on a single
# 680-frame run -- abundant multi-view coverage rejected on an arbitrary
# boundary. What Stage 3 consumes is frames that have neighbours, so that is
# what the gate measures. 400 is roughly the deep-sequence content of an area
# that comfortably passed under the old rule.
MIN_SEQUENCE_FRAMES = 5
GATE_MIN_DEEP_FRAMES = 400

# Recency. This project's output is a claim about present physical conditions:
# "this corner has no curb ramp." Assert that about a corner where a ramp was
# built in 2023 and the central claim is false, along with the project's
# credibility. Imagery age is therefore a correctness constraint, not a
# nice-to-have, and it is a hard fail rather than the warning it used to be.
#
# Median age was also the wrong statistic. An area with 2,000 frames from 2015
# and 2,000 from 2026 has a median around 2020 and is perfectly usable; an area
# with everything from 2021 has the same median and is not. What matters is
# whether *recent* imagery exists at all, so the gate now tests the age of the
# newest capture and the share of frames that are recent.
GATE_MAX_NEWEST_AGE_YEARS = 3.0
GATE_MIN_RECENT_FRACTION = 0.15
RECENT_WINDOW_YEARS = 3.0

# Retained for the staleness *warning* only; no longer a fail condition.
GATE_WARN_MEDIAN_AGE_YEARS = 6.0

# A frame needs enough resolution for a curb edge to survive rectification.
MIN_USABLE_WIDTH = 1024

# Mean bytes per 2048px JPEG. Measured over a real 50-image ingest at
# 16,189,980 bytes total (324 KB/image). The prior 450 KB placeholder
# overestimated download size by 39%.
BYTES_PER_IMAGE = 324_000


@dataclass
class SurveyResult:
    area_name: str
    bbox_param: str
    area_km2: float
    tiles_queried: int

    total_images: int = 0
    usable_images: int = 0
    pano_images: int = 0
    low_res_images: int = 0
    no_sequence_images: int = 0
    no_position_images: int = 0

    total_sequences: int = 0
    long_sequences: int = 0  # >= MIN_SEQUENCE_FRAMES
    deep_frames: int = 0  # frames living inside those sequences
    sequence_length_median: float = 0.0
    sequence_length_max: int = 0

    baseline_median_m: float | None = None
    baseline_p10_m: float | None = None
    baseline_p90_m: float | None = None
    good_baseline_fraction: float = 0.0

    capture_years: dict[int, int] = field(default_factory=dict)
    median_age_years: float | None = None
    newest_age_years: float | None = None
    recent_images: int = 0
    recent_fraction: float = 0.0
    newest_capture: str | None = None

    camera_types: dict[str, int] = field(default_factory=dict)
    images_per_km2: float = 0.0

    est_download_gb: float = 0.0

    verdict: str = "UNKNOWN"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _sequence_baselines(images_by_id: dict[str, ImageRecord],
                        image_ids: list[str]) -> list[float]:
    """Ground distance between each consecutive pair in a sequence."""
    out: list[float] = []
    prev: ImageRecord | None = None
    for iid in image_ids:
        im = images_by_id.get(iid)
        if im is None or im.best_lon is None or im.best_lat is None:
            prev = None  # gap: do not span it
            continue
        if prev is not None:
            d = haversine_m(prev.best_lon, prev.best_lat, im.best_lon, im.best_lat)
            # Filter absurd jumps: a >200 m step means the sequence has a break
            # or the GPS glitched, not a real baseline.
            if 0.0 < d < 200.0:
                out.append(d)
        prev = im
    return out


def survey_images(
    area_name: str,
    bbox: BBox,
    images: Iterable[ImageRecord],
    tile_size: float = 0.009,
) -> SurveyResult:
    """Compute coverage statistics and a go/no-go verdict."""
    res = SurveyResult(
        area_name=area_name,
        bbox_param=bbox.to_param(),
        area_km2=round(bbox.area_km2(), 3),
        tiles_queried=count_tiles(bbox, tile_size),
    )

    all_images = list(images)
    res.total_images = len(all_images)
    if not all_images:
        res.verdict = "NO-GO"
        res.reasons.append("No imagery returned for this bbox.")
        return res

    usable: list[ImageRecord] = []
    for im in all_images:
        if im.is_pano:
            res.pano_images += 1
        if im.width is not None and im.width < MIN_USABLE_WIDTH:
            res.low_res_images += 1
            continue
        if not im.sequence_id:
            res.no_sequence_images += 1
            continue
        if im.best_lon is None or im.best_lat is None:
            res.no_position_images += 1
            continue
        usable.append(im)

    res.usable_images = len(usable)
    res.images_per_km2 = round(res.usable_images / max(res.area_km2, 1e-6), 1)

    by_id = {im.id: im for im in usable}
    sequences = summarize_sequences(usable)
    res.total_sequences = len(sequences)

    lengths = [s.length for s in sequences.values()]
    if lengths:
        res.sequence_length_median = round(statistics.median(lengths), 1)
        res.sequence_length_max = max(lengths)
        long_lengths = [n for n in lengths if n >= MIN_SEQUENCE_FRAMES]
        res.long_sequences = len(long_lengths)
        res.deep_frames = sum(long_lengths)

    baselines: list[float] = []
    for seq in sequences.values():
        baselines.extend(_sequence_baselines(by_id, seq.image_ids))

    if baselines:
        res.baseline_median_m = round(statistics.median(baselines), 2)
        p10, p90 = _percentile(baselines, 0.10), _percentile(baselines, 0.90)
        res.baseline_p10_m = round(p10, 2) if p10 is not None else None
        res.baseline_p90_m = round(p90, 2) if p90 is not None else None
        good = sum(
            1 for b in baselines
            if MIN_USEFUL_BASELINE_M <= b <= MAX_USEFUL_BASELINE_M
        )
        res.good_baseline_fraction = round(good / len(baselines), 3)

    now = datetime.now(tz=timezone.utc)
    years, ages = Counter(), []
    newest: datetime | None = None
    for im in usable:
        ts = im.captured_at
        if ts is None:
            continue
        years[ts.year] += 1
        age = (now - ts).days / 365.25
        ages.append(age)
        if age <= RECENT_WINDOW_YEARS:
            res.recent_images += 1
        if newest is None or ts > newest:
            newest = ts
    res.capture_years = dict(sorted(years.items()))
    if ages:
        res.median_age_years = round(statistics.median(ages), 2)
        res.newest_age_years = round(min(ages), 2)
        res.recent_fraction = round(res.recent_images / len(ages), 3)
    if newest:
        res.newest_capture = newest.date().isoformat()

    res.camera_types = dict(Counter(im.camera_type or "unknown" for im in usable))
    res.est_download_gb = round(res.usable_images * BYTES_PER_IMAGE / 1e9, 2)

    _apply_gate(res)
    return res


def _apply_gate(res: SurveyResult) -> None:
    """Turn statistics into a decision, with the reasoning recorded."""
    fails: list[str] = []
    warns: list[str] = []

    if res.usable_images < GATE_MIN_USABLE_IMAGES:
        fails.append(
            f"Only {res.usable_images} usable images "
            f"(need >= {GATE_MIN_USABLE_IMAGES})."
        )

    if res.deep_frames < GATE_MIN_DEEP_FRAMES:
        fails.append(
            f"Only {res.deep_frames} frames inside sequences of "
            f">={MIN_SEQUENCE_FRAMES} (need >= {GATE_MIN_DEEP_FRAMES}). "
            f"Stage 3 needs frames with neighbours to triangulate against."
        )

    # -- recency: a correctness constraint for this project, not a preference --
    if res.newest_age_years is None:
        fails.append("No capture timestamps; cannot assess imagery recency.")
    else:
        if res.newest_age_years > GATE_MAX_NEWEST_AGE_YEARS:
            fails.append(
                f"Newest imagery is {res.newest_age_years} years old "
                f"(captured {res.newest_capture}; need <= "
                f"{GATE_MAX_NEWEST_AGE_YEARS:.0f}y). Barrier findings would be "
                f"claims about present conditions based on stale evidence -- a "
                f"ramp built since then makes the claim false."
            )
        elif res.recent_fraction < GATE_MIN_RECENT_FRACTION:
            fails.append(
                f"Only {res.recent_fraction:.1%} of frames are from the last "
                f"{RECENT_WINDOW_YEARS:.0f} years "
                f"(need >= {GATE_MIN_RECENT_FRACTION:.0%}). Recent coverage is "
                f"too sparse to establish current conditions across the area, "
                f"even though some recent frames exist."
            )

    if res.median_age_years is not None and \
            res.median_age_years > GATE_WARN_MEDIAN_AGE_YEARS:
        warns.append(
            f"Median imagery age {res.median_age_years} years. Recent frames "
            f"clear the gate, but most coverage is old -- verify the newest "
            f"capture per corner before reporting any barrier."
        )

    if res.baseline_median_m is None:
        fails.append("No inter-frame baselines computable; positions missing.")
    else:
        if res.good_baseline_fraction < GATE_MIN_GOOD_BASELINE_FRAC:
            fails.append(
                f"Only {res.good_baseline_fraction:.0%} of baselines fall in the "
                f"{MIN_USEFUL_BASELINE_M}-{MAX_USEFUL_BASELINE_M} m band "
                f"(need >= {GATE_MIN_GOOD_BASELINE_FRAC:.0%}). "
                f"Median spacing {res.baseline_median_m} m."
            )
        elif res.baseline_median_m > MAX_USEFUL_BASELINE_M:
            warns.append(
                f"Median baseline {res.baseline_median_m} m is long; expect "
                "matching failures on low-texture surfaces."
            )
        elif res.baseline_median_m < MIN_USEFUL_BASELINE_M:
            warns.append(
                f"Median baseline {res.baseline_median_m} m is short; "
                "triangulation will be poorly conditioned. Widen the frame "
                "window in Stage 3 rather than using adjacent pairs."
            )
        elif res.baseline_median_m > 7.0:
            warns.append(
                f"Median baseline {res.baseline_median_m} m suggests "
                "vehicle-speed capture. Car-mounted frames are shot from the "
                "roadway looking forward, so curbs sit at the oblique "
                "periphery -- adequate for detecting a ramp, harder for "
                "measuring its slope. Inspect sample frames before committing."
            )

    if res.pano_images > 0.5 * res.usable_images:
        warns.append(
            f"{res.pano_images} of {res.usable_images} frames are 360 panoramas; "
            "these need equirectangular handling before rectification."
        )

    if res.low_res_images > 0.3 * res.total_images:
        warns.append(
            f"{res.low_res_images} frames below {MIN_USABLE_WIDTH}px wide were "
            "dropped; effective coverage is thinner than the raw count suggests."
        )

    res.reasons = fails + warns
    if fails:
        res.verdict = "NO-GO"
    elif warns:
        res.verdict = "GO-WITH-CAVEATS"
    else:
        res.verdict = "GO"


def format_report(results: list[SurveyResult]) -> str:
    """Human-readable survey report for the day-1 decision log."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("REACHABLE - STAGE 0 COVERAGE SURVEY")
    lines.append(f"Generated {datetime.now(tz=timezone.utc).isoformat(timespec='seconds')}")
    lines.append("=" * 72)

    for r in results:
        lines.append("")
        lines.append(f"AREA: {r.area_name}")
        lines.append(f"  bbox            {r.bbox_param}")
        lines.append(f"  area            {r.area_km2} km^2  ({r.tiles_queried} tiles)")
        lines.append("")
        lines.append(f"  images total    {r.total_images}")
        lines.append(f"  images usable   {r.usable_images}  ({r.images_per_km2}/km^2)")
        lines.append(f"    dropped: low-res {r.low_res_images}, "
                     f"no-sequence {r.no_sequence_images}, "
                     f"no-position {r.no_position_images}")
        lines.append(f"    panoramas       {r.pano_images}")
        lines.append("")
        lines.append(f"  sequences       {r.total_sequences} "
                     f"({r.long_sequences} with >={MIN_SEQUENCE_FRAMES} frames)")
        lines.append(f"    length med/max  {r.sequence_length_median} / "
                     f"{r.sequence_length_max}")
        lines.append(f"    deep frames     {r.deep_frames}  "
                     f"(need >= {GATE_MIN_DEEP_FRAMES})")
        lines.append("")
        lines.append("  INTER-FRAME BASELINE (drives Stage 3 feasibility)")
        lines.append(f"    median          {r.baseline_median_m} m")
        lines.append(f"    p10 / p90       {r.baseline_p10_m} / {r.baseline_p90_m} m")
        lines.append(f"    in usable band  {r.good_baseline_fraction:.1%} "
                     f"({MIN_USEFUL_BASELINE_M}-{MAX_USEFUL_BASELINE_M} m)")
        lines.append("")
        lines.append("  RECENCY (drives whether findings describe today)")
        lines.append(f"    newest capture  {r.newest_capture} "
                     f"({r.newest_age_years} years ago, "
                     f"need <= {GATE_MAX_NEWEST_AGE_YEARS:.0f})")
        lines.append(f"    last {RECENT_WINDOW_YEARS:.0f}y frames  "
                     f"{r.recent_images} ({r.recent_fraction:.1%}, "
                     f"need >= {GATE_MIN_RECENT_FRACTION:.0%})")
        lines.append(f"    median age      {r.median_age_years} years")
        if r.capture_years:
            span = ", ".join(f"{y}:{n}" for y, n in r.capture_years.items())
            lines.append(f"    by year         {span}")
        lines.append("")
        lines.append(f"  camera types    {r.camera_types}")
        lines.append("")
        lines.append(f"  est. download   {r.est_download_gb} GB")
        lines.append("")
        lines.append(f"  VERDICT: {r.verdict}")
        for reason in r.reasons:
            lines.append(f"    - {reason}")
        lines.append("-" * 72)

    go = [r for r in results if r.verdict in ("GO", "GO-WITH-CAVEATS")]
    lines.append("")
    if go:
        best = max(go, key=lambda r: (r.deep_frames, r.recent_fraction))
        lines.append(f"RECOMMENDED STUDY AREA: {best.area_name}")
        lines.append(
            f"  {best.deep_frames} frames in multi-view sequences, median "
            f"baseline {best.baseline_median_m} m, "
            f"{best.recent_fraction:.1%} captured in the last "
            f"{RECENT_WINDOW_YEARS:.0f} years."
        )
    else:
        lines.append("NO AREA PASSES THE GATE.")
        stale = [r for r in results
                 if r.newest_age_years is not None
                 and r.newest_age_years > GATE_MAX_NEWEST_AGE_YEARS]
        if stale:
            lines.append(
                f"  {len(stale)} of {len(results)} areas failed on recency "
                f"alone. Crowdsourced coverage here has gone stale."
            )
        lines.append("  Action: pivot to own-capture as the primary source.")
        lines.append("  Mount a phone or action camera on a bicycle, ride the")
        lines.append("  target grid at ~2 m frame spacing, upload via the")
        lines.append("  Mapillary app. This also fixes scale recovery: a known,")
        lines.append("  measured camera height removes the single largest")
        lines.append("  source of error in Stage 3, which crowdsourced imagery")
        lines.append("  cannot give you at any density.")
        lines.append("  Historical Mapillary layers remain valuable as a")
        lines.append("  change-detection baseline against your own capture.")
    lines.append("=" * 72)
    return "\n".join(lines)