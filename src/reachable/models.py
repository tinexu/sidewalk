"""Typed records for imagery metadata.

Everything the later stages need about a frame is captured here at ingest time.
In particular `computed_geometry` and `computed_compass_angle` are Mapillary's
structure-from-motion-refined position and heading; they are meaningfully more
accurate than the raw EXIF values and are what Stage 3's GPS-baseline scale
recovery should use. Both raw and computed values are retained so the
disagreement between them can be reported as a confidence signal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

# Requested from the Graph API for every image. Keep this list in sync with
# ImageRecord.from_api. Fields cost nothing extra to request but an unknown
# field name causes the whole request to fail, so verify against live API
# responses before adding to it.
IMAGE_FIELDS: tuple[str, ...] = (
    "id",
    "sequence",
    "captured_at",
    "geometry",
    "computed_geometry",
    "compass_angle",
    "computed_compass_angle",
    "altitude",
    "computed_altitude",
    "camera_type",
    "camera_parameters",
    "width",
    "height",
    "is_pano",
    "exif_orientation",
    "thumb_1024_url",
    "thumb_2048_url",
)


def _coords(geom: Any) -> tuple[float | None, float | None]:
    """Pull (lon, lat) out of a GeoJSON Point, tolerating nulls."""
    if not isinstance(geom, dict):
        return (None, None)
    c = geom.get("coordinates")
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        return (float(c[0]), float(c[1]))
    return (None, None)


@dataclass
class ImageRecord:
    """One Mapillary frame's metadata."""

    id: str
    sequence_id: str | None
    captured_at_ms: int | None

    # Raw EXIF-derived position and heading.
    lon: float | None
    lat: float | None
    compass_angle: float | None
    altitude: float | None

    # SfM-refined position and heading. Prefer these when present.
    computed_lon: float | None
    computed_lat: float | None
    computed_compass_angle: float | None
    computed_altitude: float | None

    camera_type: str | None
    camera_parameters: list[float] | None
    width: int | None
    height: int | None
    is_pano: bool
    exif_orientation: int | None

    thumb_1024_url: str | None
    thumb_2048_url: str | None

    # Populated by the ingest layer once bytes are on disk.
    stored_key: str | None = None
    stored_bytes: int | None = None
    stored_sha256: str | None = None

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> "ImageRecord":
        lon, lat = _coords(d.get("geometry"))
        clon, clat = _coords(d.get("computed_geometry"))
        captured = d.get("captured_at")
        return cls(
            id=str(d["id"]),
            sequence_id=d.get("sequence"),
            captured_at_ms=int(captured) if captured is not None else None,
            lon=lon,
            lat=lat,
            compass_angle=d.get("compass_angle"),
            altitude=d.get("altitude"),
            computed_lon=clon,
            computed_lat=clat,
            computed_compass_angle=d.get("computed_compass_angle"),
            computed_altitude=d.get("computed_altitude"),
            camera_type=d.get("camera_type"),
            camera_parameters=d.get("camera_parameters"),
            width=d.get("width"),
            height=d.get("height"),
            is_pano=bool(d.get("is_pano", False)),
            exif_orientation=d.get("exif_orientation"),
            thumb_1024_url=d.get("thumb_1024_url"),
            thumb_2048_url=d.get("thumb_2048_url"),
        )

    @property
    def best_lon(self) -> float | None:
        return self.computed_lon if self.computed_lon is not None else self.lon

    @property
    def best_lat(self) -> float | None:
        return self.computed_lat if self.computed_lat is not None else self.lat

    @property
    def best_heading(self) -> float | None:
        if self.computed_compass_angle is not None:
            return self.computed_compass_angle
        return self.compass_angle

    @property
    def captured_at(self) -> datetime | None:
        if self.captured_at_ms is None:
            return None
        return datetime.fromtimestamp(self.captured_at_ms / 1000, tz=timezone.utc)

    def download_url(self, prefer: str = "2048") -> str | None:
        """Best available image URL.

        thumb_2048_url is adequate for curb geometry at typical capture
        distances. thumb_original_url exists but is not always granted to
        standard tokens, so it is deliberately not requested.
        """
        if prefer == "2048" and self.thumb_2048_url:
            return self.thumb_2048_url
        return self.thumb_2048_url or self.thumb_1024_url

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ImageRecord":
        return cls(**d)


@dataclass
class SequenceRecord:
    """A capture run: consecutive frames along a route.

    Sequence structure is the reason this project is possible at all. Stage 3
    recovers metric geometry from multiple views of the same corner, and
    "multiple views" means "consecutive frames in a sequence". A pipeline that
    flattens imagery into an unordered pile of photos throws away the baseline
    and cannot recover scale.
    """

    id: str
    image_ids: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.image_ids)


def summarize_sequences(images: Iterable[ImageRecord]) -> dict[str, SequenceRecord]:
    """Group images by sequence and order each by capture time.

    Images with no sequence id are dropped: a lone frame has no baseline
    partner and is useless for Stage 3.
    """
    by_seq: dict[str, list[ImageRecord]] = {}
    for im in images:
        if im.sequence_id:
            by_seq.setdefault(im.sequence_id, []).append(im)

    out: dict[str, SequenceRecord] = {}
    for sid, ims in by_seq.items():
        ims.sort(key=lambda i: (i.captured_at_ms is None, i.captured_at_ms or 0))
        out[sid] = SequenceRecord(id=sid, image_ids=[i.id for i in ims])
    return out


def write_jsonl(records: Iterable[ImageRecord], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[ImageRecord]:
    out: list[ImageRecord] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(ImageRecord.from_dict(json.loads(line)))
    return out