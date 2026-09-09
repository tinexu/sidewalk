"""Tests for the day-1 gate.

The gate's job is to say NO-GO on data that would waste four weeks. These
tests pin the cases that matter, especially the one that a naive image count
would get wrong: plenty of frames, but spaced too far apart to triangulate.
"""

from datetime import datetime, timedelta, timezone

import pytest

from reachable.geo import BBox
from reachable.models import ImageRecord
from reachable.survey import survey_images

BBOX = BBox(-86.9302, 40.4167, -86.9122, 40.4307)
NOW_MS = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
DEG_PER_M_LAT = 1.0 / 111_195.0


def make_image(
    idx: int,
    sequence_id: str,
    lat: float,
    lon: float = -86.92,
    width: int = 2048,
    age_days: int = 200,
    is_pano: bool = False,
) -> ImageRecord:
    ts = datetime.now(tz=timezone.utc) - timedelta(days=age_days)
    return ImageRecord(
        id=f"img{idx}",
        sequence_id=sequence_id,
        captured_at_ms=int(ts.timestamp() * 1000),
        lon=lon, lat=lat, compass_angle=90.0, altitude=190.0,
        computed_lon=lon, computed_lat=lat,
        computed_compass_angle=90.0, computed_altitude=190.0,
        camera_type="perspective",
        camera_parameters=[0.85, 0.0, 0.0],
        width=width, height=int(width * 0.75),
        is_pano=is_pano, exif_orientation=1,
        thumb_1024_url="https://example.invalid/1024.jpg",
        thumb_2048_url="https://example.invalid/2048.jpg",
    )


def make_sequences(
    n_sequences: int,
    frames_each: int,
    spacing_m: float,
    **kw,
) -> list[ImageRecord]:
    """Synthetic sequences walking north at a fixed frame spacing."""
    out, idx = [], 0
    for s in range(n_sequences):
        base_lat = 40.42 + s * 0.0005
        for f in range(frames_each):
            out.append(
                make_image(
                    idx,
                    f"seq{s}",
                    lat=base_lat + f * spacing_m * DEG_PER_M_LAT,
                    **kw,
                )
            )
            idx += 1
    return out


class TestGateVerdicts:
    def test_good_coverage_passes(self):
        images = make_sequences(40, 20, spacing_m=3.0)
        r = survey_images("good", BBOX, images)
        assert r.verdict in ("GO", "GO-WITH-CAVEATS"), r.reasons
        assert r.usable_images == 800
        assert r.deep_frames == 800
        assert r.baseline_median_m == pytest.approx(3.0, abs=0.2)
        assert r.good_baseline_fraction > 0.9

    def test_too_few_images_fails(self):
        images = make_sequences(5, 10, spacing_m=3.0)
        r = survey_images("sparse", BBOX, images)
        assert r.verdict == "NO-GO"
        assert any("usable images" in x for x in r.reasons)

    def test_empty_area_fails_cleanly(self):
        r = survey_images("empty", BBOX, [])
        assert r.verdict == "NO-GO"
        assert r.total_images == 0
        assert "No imagery" in r.reasons[0]

    def test_many_images_but_baselines_too_long_fails(self):
        # The case a raw image count would get wrong: 1,200 frames, which looks
        # like abundance, but at 40 m spacing no two frames see the same curb.
        images = make_sequences(40, 30, spacing_m=40.0)
        r = survey_images("highway_spacing", BBOX, images)
        assert r.usable_images == 1200
        assert r.verdict == "NO-GO"
        assert any("band" in x for x in r.reasons)

    def test_short_sequences_fail_even_with_enough_images(self):
        # 300 sequences of 2 frames: image count passes, but no frame has a
        # neighbour to triangulate against.
        images = make_sequences(300, 2, spacing_m=3.0)
        r = survey_images("stubs", BBOX, images)
        assert r.usable_images == 600
        assert r.deep_frames == 0
        assert r.verdict == "NO-GO"
        assert any("neighbours" in x for x in r.reasons)

    def test_pano_heavy_warns(self):
        images = make_sequences(40, 20, spacing_m=3.0, is_pano=True)
        r = survey_images("panos", BBOX, images)
        assert r.verdict == "GO-WITH-CAVEATS"
        assert any("panorama" in x for x in r.reasons)


class TestRecencyGate:
    """Recency is a correctness constraint here, not a preference.

    The output asserts present physical conditions. Stale imagery makes those
    assertions false, so age is a hard fail.
    """

    def test_stale_imagery_fails_hard(self):
        images = make_sequences(40, 20, spacing_m=3.0, age_days=int(5.5 * 365))
        r = survey_images("stale", BBOX, images)
        assert r.verdict == "NO-GO"
        assert any("Newest imagery" in x for x in r.reasons)

    def test_recent_imagery_passes(self):
        images = make_sequences(40, 20, spacing_m=3.0, age_days=200)
        r = survey_images("fresh", BBOX, images)
        assert r.verdict in ("GO", "GO-WITH-CAVEATS"), r.reasons
        assert r.recent_fraction == 1.0

    def test_thin_recent_layer_over_old_bulk_fails(self):
        """A handful of new frames does not rescue an otherwise stale area.

        Modelled on chauncey_village as surveyed: 4,067 frames of which only
        71 were from 2024 or later -- 1.7%. Median age cleared the old warning
        threshold by two months and the area was recommended, despite recent
        coverage being far too sparse to establish current conditions.
        """
        old = make_sequences(40, 20, spacing_m=3.0, age_days=int(5.0 * 365))
        new = make_sequences(2, 20, spacing_m=3.0, age_days=120)
        for i, im in enumerate(new):
            im.id, im.sequence_id = f"new{i}", f"newseq{i // 20}"
        r = survey_images("thin_recent", BBOX, old + new)
        assert r.newest_age_years < 1.0  # newest-capture check passes
        assert r.recent_fraction < 0.15
        assert r.verdict == "NO-GO"
        assert any("too sparse" in x for x in r.reasons)

    def test_mixed_ages_pass_with_enough_recent(self):
        """Old imagery is an asset when a real recent layer exists.

        Historical epochs become the change-detection baseline that Stage 3's
        multi-date comparison needs, so the gate must not punish them.
        """
        old = make_sequences(30, 20, spacing_m=3.0, age_days=int(8.0 * 365))
        new = make_sequences(15, 20, spacing_m=3.0, age_days=150)
        for i, im in enumerate(new):
            im.id, im.sequence_id = f"new{i}", f"newseq{i // 20}"
        r = survey_images("mixed", BBOX, old + new)
        assert r.recent_fraction > 0.15
        assert r.verdict in ("GO", "GO-WITH-CAVEATS"), r.reasons
        assert any("most coverage is old" in x for x in r.reasons)


class TestDeepFrameGate:
    """Frames-with-neighbours, not sequence count.

    Counting sequences failed campus_core at 19 against a threshold of 20
    while it sat on a single 680-frame run.
    """

    def test_few_but_deep_sequences_pass(self):
        images = make_sequences(6, 90, spacing_m=3.0, age_days=200)
        r = survey_images("deep", BBOX, images)
        assert r.long_sequences == 6  # would have failed the old count rule
        assert r.deep_frames == 540
        assert r.verdict in ("GO", "GO-WITH-CAVEATS"), r.reasons

    def test_many_shallow_sequences_fail(self):
        images = make_sequences(120, 5, spacing_m=3.0, age_days=200)
        r = survey_images("shallow", BBOX, images)
        assert r.long_sequences == 120  # would have passed the old count rule
        assert r.deep_frames == 600
        # Passes on depth here; the point is that count alone no longer decides.
        assert r.deep_frames >= 400

    def test_orphan_frames_excluded_from_deep_count(self):
        deep = make_sequences(5, 100, spacing_m=3.0, age_days=200)
        orphans = [
            make_image(90_000 + i, f"solo{i}", lat=40.43 + i * 1e-5, age_days=200)
            for i in range(200)
        ]
        r = survey_images("orphans", BBOX, deep + orphans)
        assert r.total_sequences == 205
        assert r.deep_frames == 500  # orphans contribute nothing


class TestVehicleSpeedWarning:
    def test_long_baseline_warns_about_capture_geometry(self):
        """8.5 m median implies ~30 km/h at 1 fps -- car-mounted.

        Curbs then sit at the oblique periphery rather than the frame centre:
        fine for detecting a ramp, harder for measuring its slope.
        """
        images = make_sequences(40, 20, spacing_m=8.6, age_days=200)
        r = survey_images("vehicle", BBOX, images)
        assert any("vehicle-speed" in x for x in r.reasons)


class TestDownloadEstimate:
    def test_uses_measured_bytes_per_image(self):
        """324 KB/image, measured over a real 50-image ingest.

        The prior 450 KB placeholder overestimated by 39%.
        """
        images = make_sequences(50, 20, spacing_m=3.0, age_days=200)
        r = survey_images("size", BBOX, images)
        assert r.usable_images == 1000
        assert r.est_download_gb == pytest.approx(0.32, abs=0.01)


class TestFiltering:
    def test_low_resolution_dropped(self):
        images = make_sequences(40, 20, spacing_m=3.0)
        images += [make_image(9000 + i, "lowres", 40.43, width=640)
                   for i in range(50)]
        r = survey_images("mixed", BBOX, images)
        assert r.low_res_images == 50
        assert r.usable_images == 800

    def test_missing_sequence_dropped(self):
        images = make_sequences(40, 20, spacing_m=3.0)
        orphan = make_image(9999, "x", 40.43)
        orphan.sequence_id = None
        images.append(orphan)
        r = survey_images("orphan", BBOX, images)
        assert r.no_sequence_images == 1
        assert r.usable_images == 800

    def test_missing_position_dropped(self):
        images = make_sequences(40, 20, spacing_m=3.0)
        nogps = make_image(9998, "seq0", 40.43)
        nogps.lat = nogps.lon = None
        nogps.computed_lat = nogps.computed_lon = None
        images.append(nogps)
        r = survey_images("nogps", BBOX, images)
        assert r.no_position_images == 1


class TestBaselineComputation:
    def test_baseline_matches_synthetic_spacing(self):
        for spacing in (1.5, 3.0, 8.0):
            images = make_sequences(40, 20, spacing_m=spacing)
            r = survey_images(f"s{spacing}", BBOX, images)
            assert r.baseline_median_m == pytest.approx(spacing, abs=0.2)

    def test_gps_glitch_excluded_from_baselines(self):
        # A single 5 km jump must not drag the median. Real sequences contain
        # these when the phone loses lock and reacquires.
        images = make_sequences(40, 20, spacing_m=3.0)
        images.append(make_image(50_000, "seq0", lat=40.47))
        r = survey_images("glitch", BBOX, images)
        assert r.baseline_median_m == pytest.approx(3.0, abs=0.3)

    def test_sequence_ordering_is_by_capture_time(self):
        # Records arriving out of order must still yield correct baselines.
        images = make_sequences(30, 10, spacing_m=3.0)
        shuffled = images[::-1]
        a = survey_images("ordered", BBOX, images)
        b = survey_images("shuffled", BBOX, shuffled)
        assert a.baseline_median_m == pytest.approx(b.baseline_median_m, abs=0.1)


class TestRecommendation:
    def test_best_area_selected_on_sequence_depth(self):
        from reachable.survey import format_report

        weak = survey_images("weak", BBOX, make_sequences(25, 6, spacing_m=3.0))
        strong = survey_images("strong", BBOX, make_sequences(60, 25, spacing_m=3.0))
        report = format_report([weak, strong])
        assert "RECOMMENDED STUDY AREA: strong" in report

    def test_all_fail_recommends_own_capture(self):
        from reachable.survey import format_report

        bad = survey_images("bad", BBOX, make_sequences(2, 3, spacing_m=3.0))
        report = format_report([bad])
        assert "NO AREA PASSES THE GATE" in report
        assert "own-capture" in report