"""Geographic primitives and bbox tiling.

The Mapillary Graph API constrains bbox queries against /images, /map_features
and detection search to be strictly smaller than 0.01 degrees square (formalised
2026-01-16). Any study area of useful size therefore has to be decomposed into
a grid of compliant tiles before querying.

This module owns that decomposition and nothing else, so the tiling rule can be
unit-tested without touching the network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterator

# Mapillary's hard limit. We tile below it with a safety margin because the API
# rejects the request outright rather than clamping it, and a rejected tile in
# the middle of a long ingest run is expensive to recover from.
MAX_BBOX_DEGREES = 0.01
DEFAULT_TILE_DEGREES = 0.009

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class BBox:
    """A geographic bounding box in WGS84 degrees.

    Field order matches the Mapillary API's expected `bbox` parameter:
    west, south, east, north.
    """

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if self.west >= self.east:
            raise ValueError(f"west ({self.west}) must be < east ({self.east})")
        if self.south >= self.north:
            raise ValueError(f"south ({self.south}) must be < north ({self.north})")
        if not (-180 <= self.west < self.east <= 180):
            raise ValueError("longitudes out of range")
        if not (-90 <= self.south < self.north <= 90):
            raise ValueError("latitudes out of range")

    @property
    def width(self) -> float:
        return self.east - self.west

    @property
    def height(self) -> float:
        return self.north - self.south

    @property
    def center(self) -> tuple[float, float]:
        """(lon, lat) of the box centre."""
        return (self.west + self.width / 2, self.south + self.height / 2)

    def is_api_compliant(self, limit: float = MAX_BBOX_DEGREES) -> bool:
        """True if this box can be sent to the Graph API without tiling."""
        return self.width < limit and self.height < limit

    def to_param(self) -> str:
        """Serialise for the API's `bbox` query parameter."""
        return f"{self.west},{self.south},{self.east},{self.north}"

    def contains(self, lon: float, lat: float) -> bool:
        return self.west <= lon <= self.east and self.south <= lat <= self.north

    def area_km2(self) -> float:
        """Approximate area, using the mean latitude for longitude foreshortening."""
        mean_lat_rad = math.radians((self.north + self.south) / 2)
        m_per_deg_lat = (math.pi / 180) * EARTH_RADIUS_M
        m_per_deg_lon = m_per_deg_lat * math.cos(mean_lat_rad)
        return (self.height * m_per_deg_lat) * (self.width * m_per_deg_lon) / 1e6

    @classmethod
    def from_param(cls, s: str) -> "BBox":
        parts = [float(p) for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError(f"expected 4 comma-separated values, got {len(parts)}")
        return cls(*parts)

    @classmethod
    def from_center(cls, lon: float, lat: float, radius_m: float) -> "BBox":
        """Square box of the given half-width in metres around a point."""
        m_per_deg_lat = (math.pi / 180) * EARTH_RADIUS_M
        m_per_deg_lon = m_per_deg_lat * math.cos(math.radians(lat))
        d_lat = radius_m / m_per_deg_lat
        d_lon = radius_m / m_per_deg_lon
        return cls(lon - d_lon, lat - d_lat, lon + d_lon, lat + d_lat)


# Degree extents are differences of numbers near 87, so they carry float error
# around 1e-13. Without a tolerance, a span that is exactly 2 tiles wide
# computes as 2.000000000000076 and ceils to 3, producing a degenerate third
# column. The tolerance is far larger than the error and far smaller than any
# meaningful fraction of a tile.
_STEP_EPS = 1e-9


def _n_steps(extent: float, step: float) -> int:
    """Number of tiles needed to span `extent`, tolerant of float drift."""
    return max(1, math.ceil(extent / step - _STEP_EPS))


def tile_bbox(
    bbox: BBox, tile_size: float = DEFAULT_TILE_DEGREES
) -> Iterator[BBox]:
    """Split a bbox into a grid of API-compliant tiles.

    Tiles are emitted in row-major order from the south-west corner.

    The final column and row are snapped to the parent's east and north edges
    rather than computed as `west + tile_size`. This is not cosmetic: repeated
    addition drifts downward (0.018 + 0.009 == 0.026999999999999996), so a
    computed final edge falls *short* of the parent and leaves an unqueried
    sliver along two sides of every study area. Snapping guarantees full
    coverage.

    Adjacent tiles share an edge, so an image sitting exactly on a boundary can
    be returned by two tiles. Deduplication by image id happens downstream in
    the ingest layer, which is the right place for it -- the tiler stays pure.
    """
    if tile_size >= MAX_BBOX_DEGREES:
        raise ValueError(
            f"tile_size {tile_size} must be < {MAX_BBOX_DEGREES} "
            "to satisfy the Mapillary bbox constraint"
        )

    n_cols = _n_steps(bbox.width, tile_size)
    n_rows = _n_steps(bbox.height, tile_size)

    for row in range(n_rows):
        south = bbox.south + row * tile_size
        north = bbox.north if row == n_rows - 1 else south + tile_size
        if north <= south:
            continue
        for col in range(n_cols):
            west = bbox.west + col * tile_size
            east = bbox.east if col == n_cols - 1 else west + tile_size
            if east <= west:
                continue
            tile = BBox(west, south, east, north)
            # Defensive: a snapped edge must never push a tile over the API
            # limit. Failing here is far cheaper than a 400 mid-ingest.
            if not tile.is_api_compliant():
                raise AssertionError(
                    f"tiler produced non-compliant tile {tile.to_param()} "
                    f"({tile.width:.6f} x {tile.height:.6f} deg)"
                )
            yield tile


def count_tiles(bbox: BBox, tile_size: float = DEFAULT_TILE_DEGREES) -> int:
    """Tile count without materialising the tiles. Used for cost estimation."""
    return _n_steps(bbox.width, tile_size) * _n_steps(bbox.height, tile_size)


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres.

    Used for GPS-baseline scale recovery in Stage 3 and for sequence-gap
    detection here in Stage 0.
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))