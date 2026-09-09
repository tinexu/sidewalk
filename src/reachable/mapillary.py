"""Mapillary Graph API client.

Scope is deliberately narrow: list image metadata within a bbox, and fetch
image bytes. Everything else the project needs is derived locally.

Licensing obligations that follow from using this client, and which must be
honoured in the UI and report:
  - Imagery is CC BY-SA 4.0. Derived products carrying image content inherit
    share-alike terms.
  - Integrating data extracted through the API or vector tiles requires
    visibly displaying the Mapillary logo with a link back to mapillary.com.
See LICENSES.md at the repository root.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Iterator

import requests

from .geo import BBox, tile_bbox
from .models import IMAGE_FIELDS, ImageRecord

log = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.mapillary.com"
DEFAULT_LIMIT = 2000
DEFAULT_TIMEOUT = 30

# Retried. 429 is rate limiting; 5xx are transient upstream failures.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class MapillaryError(RuntimeError):
    """Non-retryable API failure."""


class RateLimiter:
    """Token-bucket-ish limiter: at most `rate` calls per second, thread-safe.

    Mapillary's published limits vary by usage tier, so this is set
    conservatively by default. A survey of a few square kilometres is hundreds
    of tile queries; getting throttled halfway through and having to restart is
    a worse outcome than running slightly slower.
    """

    def __init__(self, rate: float = 8.0) -> None:
        self._min_interval = 1.0 / rate if rate > 0 else 0.0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._last + self._min_interval - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self._last = now


class MapillaryClient:
    def __init__(
        self,
        token: str | None = None,
        rate: float = 8.0,
        max_retries: int = 5,
        timeout: int = DEFAULT_TIMEOUT,
        base_url: str = GRAPH_BASE,
    ) -> None:
        self.token = token or os.environ.get("MAPILLARY_TOKEN")
        if not self.token:
            raise MapillaryError(
                "No API token. Set MAPILLARY_TOKEN or pass token=. "
                "Create one at https://www.mapillary.com/dashboard/developers"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = RateLimiter(rate)

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"OAuth {self.token}",
                "User-Agent": "reachable-sidewalk-audit/0.1 (OpenCV5 competition)",
            }
        )

        self.request_count = 0
        self.retry_count = 0

    # -- low level ---------------------------------------------------------

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                self.request_count += 1
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                self.retry_count += 1
                self._backoff(attempt)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in RETRY_STATUS:
                self.retry_count += 1
                # Honour Retry-After when the server sends one.
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(min(float(retry_after), 60.0))
                        continue
                    except ValueError:
                        pass
                self._backoff(attempt)
                continue

            if resp.status_code in (401, 403):
                raise MapillaryError(
                    f"Auth failed ({resp.status_code}). Check MAPILLARY_TOKEN "
                    f"scopes. Body: {resp.text[:300]}"
                )
            if resp.status_code == 400:
                raise MapillaryError(
                    f"Bad request ({resp.status_code}). Most often an oversized "
                    f"bbox or an unrecognised field name. Body: {resp.text[:300]}"
                )
            raise MapillaryError(
                f"HTTP {resp.status_code} from {url}. Body: {resp.text[:300]}"
            )

        raise MapillaryError(
            f"Exhausted {self.max_retries} retries for {url}"
        ) from last_exc

    def _backoff(self, attempt: int) -> None:
        # Exponential with jitter, capped. Jitter matters when several tile
        # workers get throttled simultaneously and would otherwise retry in
        # lockstep.
        delay = min(2.0**attempt, 30.0) * (0.5 + random.random())
        log.debug("backing off %.1fs (attempt %d)", delay, attempt + 1)
        time.sleep(delay)

    # -- metadata ----------------------------------------------------------

    def images_in_tile(
        self, tile: BBox, limit: int = DEFAULT_LIMIT
    ) -> Iterator[ImageRecord]:
        """All images within one API-compliant tile, following pagination.

        Raises if the tile is too large, rather than letting the API reject it
        with an opaque 400.
        """
        if not tile.is_api_compliant():
            raise MapillaryError(
                f"Tile {tile.to_param()} is {tile.width:.5f}x{tile.height:.5f} deg; "
                "must be < 0.01 deg square. Use geo.tile_bbox()."
            )

        url = f"{self.base_url}/images"
        params: dict[str, Any] | None = {
            "fields": ",".join(IMAGE_FIELDS),
            "bbox": tile.to_param(),
            "limit": limit,
        }

        pages = 0
        while url:
            payload = self._get(url, params)
            for item in payload.get("data", []):
                try:
                    yield ImageRecord.from_api(item)
                except (KeyError, TypeError, ValueError) as exc:
                    log.warning("skipping malformed image record: %s", exc)

            pages += 1
            nxt = (payload.get("paging") or {}).get("next")
            # The `next` URL carries its own querystring, including the token.
            url, params = (nxt, None) if nxt else (None, None)

            if pages > 100:
                log.warning(
                    "tile %s exceeded 100 pages; truncating", tile.to_param()
                )
                break

    def images_in_bbox(
        self,
        bbox: BBox,
        tile_size: float = 0.009,
        progress: bool = True,
    ) -> Iterator[ImageRecord]:
        """All images in an arbitrarily large bbox, deduplicated by id.

        Tiles share edges, so a frame on a boundary can be returned twice.
        Dedup happens here so the tiler stays a pure function.
        """
        seen: set[str] = set()
        tiles = list(tile_bbox(bbox, tile_size))
        for i, tile in enumerate(tiles, 1):
            if progress:
                log.info("tile %d/%d  %s", i, len(tiles), tile.to_param())
            for rec in self.images_in_tile(tile):
                if rec.id in seen:
                    continue
                seen.add(rec.id)
                yield rec

    # -- bytes -------------------------------------------------------------

    def fetch_image_bytes(self, url: str) -> bytes:
        """Download image bytes from a thumb URL.

        Thumb URLs are pre-signed CDN links and must not carry the OAuth
        header, so this uses a bare request rather than the authed session.
        They also expire, which is why ingest fetches metadata and bytes in
        the same pass rather than caching URLs for later.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            self.limiter.wait()
            try:
                self.request_count += 1
                resp = requests.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                self.retry_count += 1
                self._backoff(attempt)
                continue

            if resp.status_code == 200:
                return resp.content
            if resp.status_code in RETRY_STATUS:
                self.retry_count += 1
                self._backoff(attempt)
                continue
            raise MapillaryError(f"HTTP {resp.status_code} fetching image bytes")

        raise MapillaryError("Exhausted retries fetching image bytes") from last_exc

    def stats(self) -> dict[str, int]:
        return {"requests": self.request_count, "retries": self.retry_count}