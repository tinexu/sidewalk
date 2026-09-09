"""Command line interface for Stage 0.

    reachable survey --areas config/areas.json --out reports/
    reachable ingest --area campus_north --storage s3://my-bucket/reachable
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .geo import BBox, count_tiles
from .ingest import Ingestor
from .mapillary import MapillaryClient, MapillaryError
from .storage import make_storage
from .survey import format_report, survey_images


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Requests' connection chatter drowns everything else at DEBUG.
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _load_areas(path: str) -> dict[str, BBox]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    return {name: BBox.from_param(bbox) for name, bbox in raw["areas"].items()}


def cmd_survey(args: argparse.Namespace) -> int:
    areas = _load_areas(args.areas)
    if args.area:
        areas = {k: v for k, v in areas.items() if k in args.area}
        if not areas:
            print(f"No matching areas in {args.areas}", file=sys.stderr)
            return 2

    total_tiles = sum(count_tiles(b, args.tile_size) for b in areas.values())
    print(f"Surveying {len(areas)} area(s), {total_tiles} tiles total.")
    print("Metadata only, no image downloads.\n")

    client = MapillaryClient(rate=args.rate)
    results = []
    for name, bbox in areas.items():
        print(f"-> {name} ({count_tiles(bbox, args.tile_size)} tiles)")
        images = list(client.images_in_bbox(bbox, tile_size=args.tile_size))
        results.append(survey_images(name, bbox, images, tile_size=args.tile_size))

    report = format_report(results)
    print("\n" + report)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "coverage_survey.txt").write_text(report, encoding="utf-8")
    (out / "coverage_survey.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8"
    )
    print(f"\nWritten to {out}/coverage_survey.{{txt,json}}")

    # Non-zero exit if nothing passed, so CI or a wrapper script can gate on it.
    return 0 if any(r.verdict != "NO-GO" for r in results) else 1


def cmd_ingest(args: argparse.Namespace) -> int:
    areas = _load_areas(args.areas)
    if args.area not in areas:
        print(f"Unknown area '{args.area}'. Known: {sorted(areas)}", file=sys.stderr)
        return 2
    bbox = areas[args.area]

    client = MapillaryClient(rate=args.rate)
    storage = make_storage(args.storage)
    ingestor = Ingestor(
        client=client,
        storage=storage,
        workers=args.workers,
        skip_panos=args.skip_panos,
        min_width=args.min_width,
    )

    manifest = ingestor.run(
        area=args.area,
        bbox=bbox,
        tile_size=args.tile_size,
        limit=args.limit,
        local_metadata_path=args.metadata_out,
    )

    print(json.dumps(manifest["counts"], indent=2))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Dry run: tile counts and cost estimate without touching the network."""
    areas = _load_areas(args.areas)
    print(f"{'area':<24} {'km^2':>8} {'tiles':>7}")
    print("-" * 42)
    total = 0
    for name, bbox in areas.items():
        n = count_tiles(bbox, args.tile_size)
        total += n
        print(f"{name:<24} {bbox.area_km2():>8.2f} {n:>7}")
    print("-" * 42)
    print(f"{'TOTAL':<24} {'':>8} {total:>7}")
    print(f"\nAt {args.rate} req/s, metadata survey takes ~{total / args.rate / 60:.1f} min")
    print("(before pagination, which multiplies request count on dense tiles)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    # Common flags live on a parent parser so they can be written after the
    # subcommand ("reachable plan --areas x.json"), which is what everyone
    # types. Declaring them only on the top-level parser forces them before
    # the subcommand and produces a confusing "unrecognized arguments" error.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true")
    common.add_argument("--areas", default="config/areas.json")
    common.add_argument("--tile-size", type=float, default=0.009,
                        help="degrees; must be < 0.01 per API constraint")
    common.add_argument("--rate", type=float, default=8.0,
                        help="API requests/sec")

    p = argparse.ArgumentParser(prog="reachable", description=__doc__,
                                parents=[common])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("survey", parents=[common],
                       help="coverage survey and go/no-go gate")
    s.add_argument("--area", action="append", help="limit to named area(s)")
    s.add_argument("--out", default="reports")
    s.set_defaults(func=cmd_survey)

    i = sub.add_parser("ingest", parents=[common],
                       help="download and cache imagery")
    i.add_argument("--area", required=True)
    i.add_argument("--storage", required=True,
                   help="s3://bucket/prefix or a local path")
    i.add_argument("--workers", type=int, default=8)
    i.add_argument("--limit", type=int, help="cap images, for smoke tests")
    i.add_argument("--skip-panos", action="store_true")
    i.add_argument("--min-width", type=int, default=1024)
    i.add_argument("--metadata-out", help="also write a local JSONL copy")
    i.set_defaults(func=cmd_ingest)

    pl = sub.add_parser("plan", parents=[common],
                        help="tile/cost estimate, no network")
    pl.set_defaults(func=cmd_plan)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        return args.func(args)
    except MapillaryError as exc:
        print(f"\nMapillary API error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted. Ingest is resumable; rerun the same command.",
              file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())