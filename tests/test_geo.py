"""Tests for bbox tiling.

The 0.01-degree constraint is a hard API limit that rejects requests rather
than clamping them, so a tiling bug does not degrade gracefully -- it fails the
whole ingest run. These tests exist to make that failure mode impossible.
"""

import math

import pytest

from reachable.geo import (
    MAX_BBOX_DEGREES,
    BBox,
    count_tiles,
    haversine_m,
    tile_bbox,
)


class TestBBox:
    def test_rejects_inverted_longitude(self):
        with pytest.raises(ValueError, match="west"):
            BBox(-86.90, 40.41, -86.93, 40.43)

    def test_rejects_inverted_latitude(self):
        with pytest.raises(ValueError, match="south"):
            BBox(-86.93, 40.43, -86.90, 40.41)

    def test_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            BBox(-181.0, 40.41, -86.90, 40.43)

    def test_param_roundtrip(self):
        b = BBox(-86.9302, 40.4167, -86.9122, 40.4307)
        assert BBox.from_param(b.to_param()) == b

    def test_param_order_is_wsen(self):
        # The API expects west,south,east,north. Getting this order wrong
        # returns imagery from the wrong place rather than erroring.
        b = BBox(-86.93, 40.41, -86.91, 40.43)
        assert b.to_param() == "-86.93,40.41,-86.91,40.43"

    def test_compliance_check(self):
        assert BBox(0, 0, 0.009, 0.009).is_api_compliant()
        assert not BBox(0, 0, 0.02, 0.009).is_api_compliant()
        assert not BBox(0, 0, 0.009, 0.02).is_api_compliant()
        # Exactly at the limit must fail: the constraint is strict inequality.
        assert not BBox(0, 0, MAX_BBOX_DEGREES, 0.005).is_api_compliant()

    def test_from_center_is_approximately_square_in_metres(self):
        b = BBox.from_center(-86.92, 40.42, 750.0)
        # 750 m half-width means ~1.5 km on a side -> ~2.25 km^2
        assert 2.0 < b.area_km2() < 2.5
        # Longitude span must exceed latitude span at this latitude.
        assert b.width > b.height

    def test_area_accounts_for_latitude(self):
        equator = BBox(0, 0, 0.01, 0.01)
        northern = BBox(0, 60, 0.01, 60.01)
        # Same degree extent, but far less ground area at 60 N.
        assert northern.area_km2() < equator.area_km2() * 0.55


class TestTiling:
    def test_every_tile_is_api_compliant(self):
        bbox = BBox(-86.9302, 40.4167, -86.9122, 40.4307)
        tiles = list(tile_bbox(bbox))
        assert tiles
        for t in tiles:
            assert t.is_api_compliant(), f"{t.to_param()} violates the bbox limit"

    def test_tiles_cover_the_parent_exactly(self):
        bbox = BBox(-86.9302, 40.4167, -86.9122, 40.4307)
        tiles = list(tile_bbox(bbox))
        assert min(t.west for t in tiles) == pytest.approx(bbox.west)
        assert min(t.south for t in tiles) == pytest.approx(bbox.south)
        assert max(t.east for t in tiles) == pytest.approx(bbox.east)
        assert max(t.north for t in tiles) == pytest.approx(bbox.north)

    def test_tiles_do_not_overhang_the_parent(self):
        # Overhang would fetch imagery outside the study area and inflate cost.
        bbox = BBox(0, 0, 0.025, 0.025)  # not a whole multiple of tile size
        for t in tile_bbox(bbox):
            assert t.east <= bbox.east + 1e-12
            assert t.north <= bbox.north + 1e-12

    def test_tile_area_sums_to_parent_area(self):
        bbox = BBox(0, 40, 0.025, 40.025)
        total = sum(t.area_km2() for t in tile_bbox(bbox))
        assert total == pytest.approx(bbox.area_km2(), rel=1e-6)

    def test_small_bbox_yields_single_tile(self):
        bbox = BBox(0, 0, 0.005, 0.005)
        assert len(list(tile_bbox(bbox))) == 1

    def test_count_matches_materialised_tiles(self):
        for bbox in [
            BBox(0, 0, 0.005, 0.005),
            BBox(0, 0, 0.025, 0.014),
            BBox(-86.9302, 40.4167, -86.9122, 40.4307),
        ]:
            assert count_tiles(bbox) == len(list(tile_bbox(bbox)))

    def test_rejects_noncompliant_tile_size(self):
        bbox = BBox(0, 0, 0.05, 0.05)
        with pytest.raises(ValueError, match="0.01"):
            list(tile_bbox(bbox, tile_size=0.01))
        with pytest.raises(ValueError):
            list(tile_bbox(bbox, tile_size=0.02))

    def test_final_edges_reach_the_parent_exactly(self):
        """Regression: repeated addition drifts downward.

        0.018 + 0.009 == 0.026999999999999996, so a final edge computed as
        west + tile_size falls short of the parent's east edge and leaves an
        unqueried sliver along two sides of every study area. Silent, and it
        would have shown up as mysteriously missing corners in Stage 4.
        """
        bbox = BBox(0, 0, 0.027, 0.018)
        tiles = list(tile_bbox(bbox))
        assert max(t.east for t in tiles) == bbox.east
        assert max(t.north for t in tiles) == bbox.north

    def test_no_degenerate_tile_from_float_error(self):
        """Regression: 0.018000000000000682 / 0.009 ceils to 3, not 2.

        Real-world coordinate spans are differences of numbers near 87, so the
        exact-multiple case is never exact. Without tolerance the tiler emits a
        zero-width third column and count_tiles disagrees with reality.
        """
        bbox = BBox(-86.9302, 40.4167, -86.9122, 40.4307)
        assert bbox.width != 0.018  # the premise: drift is present
        tiles = list(tile_bbox(bbox))
        assert len(tiles) == count_tiles(bbox)
        for t in tiles:
            assert t.width > 0 and t.height > 0

    def test_all_configured_areas_tile_cleanly(self):
        """Every bbox actually shipped in config must tile without surprises."""
        import json
        from pathlib import Path

        cfg = Path(__file__).resolve().parents[1] / "config" / "areas.json"
        areas = json.loads(cfg.read_text())["areas"]
        assert areas
        for name, param in areas.items():
            bbox = BBox.from_param(param)
            tiles = list(tile_bbox(bbox))
            assert tiles, name
            assert len(tiles) == count_tiles(bbox), name
            assert max(t.east for t in tiles) == bbox.east, name
            assert max(t.north for t in tiles) == bbox.north, name
            for t in tiles:
                assert t.is_api_compliant(), f"{name}: {t.to_param()}"

    def test_no_gaps_between_adjacent_tiles(self):
        bbox = BBox(0, 0, 0.027, 0.018)
        tiles = list(tile_bbox(bbox))
        # Sample a dense grid of points; every one must land inside some tile.
        for i in range(15):
            for j in range(15):
                lon = bbox.west + bbox.width * i / 14
                lat = bbox.south + bbox.height * j / 14
                assert any(t.contains(lon, lat) for t in tiles), (lon, lat)


class TestHaversine:
    def test_zero_distance(self):
        assert haversine_m(-86.92, 40.42, -86.92, 40.42) == pytest.approx(0.0)

    def test_known_latitude_degree(self):
        # One degree of latitude is close to 111.2 km everywhere.
        d = haversine_m(0, 40.0, 0, 41.0)
        assert d == pytest.approx(111_195, rel=0.01)

    def test_typical_capture_baseline(self):
        # A few metres apart, the scale that matters for Stage 3 baselines.
        d = haversine_m(-86.92, 40.42, -86.92, 40.42 + 0.000027)
        assert 2.5 < d < 3.5

    def test_symmetric(self):
        a = haversine_m(-86.92, 40.42, -86.90, 40.44)
        b = haversine_m(-86.90, 40.44, -86.92, 40.42)
        assert a == pytest.approx(b)

    def test_matches_flat_earth_approximation_at_short_range(self):
        # Sanity check against simple planar geometry over ~100 m.
        lat = 40.42
        dlat, dlon = 0.0005, 0.0005
        d = haversine_m(-86.92, lat, -86.92 + dlon, lat + dlat)
        m_lat = dlat * 111_195
        m_lon = dlon * 111_195 * math.cos(math.radians(lat))
        assert d == pytest.approx(math.hypot(m_lat, m_lon), rel=0.01)