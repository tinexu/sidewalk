# Reachable — Stage 0: imagery acquisition and caching 


Fetches Mapillary street-level imagery for a study area, verifies it, and
caches it partitioned by sequence for Stage 1 onward.

Covers days 1–4 of the build plan (Aug 26–29): coverage survey, the go/no-go
gate, and the ingest pipeline.

## What this does

| Command | Purpose |
|---|---|
| `plan` | Tile and cost estimate. No network, no token. |
| `survey` | Metadata-only coverage survey over candidate areas, with a go/no-go verdict. **The day-1 gate.** |
| `ingest` | Download, verify and cache imagery plus metadata for one area. |

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Get an API token from the [Mapillary developer
dashboard](https://www.mapillary.com/dashboard/developers) — free, and the API
is free to use under CC BY-SA 4.0.

```bash
export MAPILLARY_TOKEN='MLY|...'
```

## Run it

```bash
# 1. Estimate the work. No network.
PYTHONPATH=src python3 -m reachable.cli plan

# 2. Prove the pipeline works with no token at all.
PYTHONPATH=src python3 scripts/demo_offline.py

# 3. THE DAY-1 GATE. Metadata only, no image downloads.
PYTHONPATH=src python3 -m reachable.cli survey --out reports/

# 4. Ingest the winning area. Resumable — just rerun if it drops.
PYTHONPATH=src python3 -m reachable.cli ingest \
    --area wl_residential_north \
    --storage ./data/cache \
    --limit 50            # drop --limit for the real run

# Same command against S3 in the Batch pipeline:
PYTHONPATH=src python3 -m reachable.cli ingest \
    --area wl_residential_north \
    --storage s3://reachable-imagery/v1
```

Tests:

```bash
PYTHONPATH=src python3 -m pytest tests/ -q     # 62 tests, no network needed
```

## Read the survey output correctly

The verdict does **not** turn on image count. It turns on **median inter-frame
baseline** — the ground distance between consecutive frames in a sequence.

Stage 3 recovers metric geometry by triangulating between frames. That needs a
baseline in a usable band:

- **under ~1 m** — triangulation is ill-conditioned, depth error explodes
- **over ~10 m** — viewpoint change defeats descriptor matching, and the same
  curb may not appear in both frames at all

An area with 5,000 frames at 25 m spacing is worse than one with 800 frames at
3 m. The survey will fail the first and pass the second, and that is correct.
`scripts/demo_offline.py` demonstrates exactly this discrimination.

If everything comes back NO-GO, the plan is already written: mount a phone or
action camera on a bicycle, ride the target grid at roughly 2 m frame spacing,
upload through the Mapillary app. One afternoon beats every surveyed area, and
the imagery stays open for the community afterwards.

## Verify these against the live API on day 1

**Everything here was developed and tested against a mock**, because the
development sandbox could not reach `graph.mapillary.com`. The pipeline logic
— tiling, pagination, dedup, resumability, corruption rejection, manifests —
is covered by 62 passing tests. What a mock cannot verify is the real API's
surface. Check these first, in this order:

1. **Field names.** `models.IMAGE_FIELDS` lists what is requested. An
   unrecognised field fails the *entire* request with a 400, not just that
   field. Start with a single tile and a trimmed field list, then add back.

   ```bash
   curl -s -H "Authorization: OAuth $MAPILLARY_TOKEN" \
     "https://graph.mapillary.com/images?fields=id,sequence,computed_geometry&bbox=-86.9302,40.4167,-86.9212,40.4257&limit=5" | head -40
   ```

2. **Auth header form.** This client sends `Authorization: OAuth <token>`. If
   that 401s, try `?access_token=<token>` as a query parameter instead.

3. **The 0.01-degree bbox limit.** Confirm it still rejects at 0.01 and accepts
   at 0.009. The constraint was formalised in January 2026; if it has changed,
   update `geo.MAX_BBOX_DEGREES` and the tests will tell you what else moves.

4. **Pagination shape.** This client follows `paging.next`. Verify against a
   dense tile that actually paginates.

5. **Rate limits.** Default is 8 req/s, set conservatively. Watch for 429s on
   the first survey and adjust `--rate`.

6. **Thumb URL expiry.** These are pre-signed CDN links, which is why metadata
   and bytes are fetched in one pass. If a long ingest starts returning 403s on
   bytes, the URLs are expiring faster than the run takes — reduce batch size.

7. **Real file sizes.** `survey.py` estimates 450 KB per 2048px frame. Measure
   after the first ingest and update the constant.

## Layout

```
imagery/seq={sequence_id}/{image_id}.jpg
metadata/seq={sequence_id}/images.jsonl     # ordered by capture time
manifests/{area}/manifest.json
```

Partitioned by **sequence**, not geography. Stage 3 reads whole sequences to
build baselines, so this gives contiguous reads and lets AWS Batch shard by
sequence with no cross-shard coordination. Do not reorganise this by tile or
bbox without reworking Stage 3's read path.

## Design decisions worth knowing

**Ingest is resumable.** Every image is checked against storage before
fetching. A dropped connection costs the remainder of one batch, not the run.
Rerun the identical command.

**Downloads are decode-verified before they are cached.** A truncated file that
lands in the cache is worse than no file, because every later run treats it as
a hit. `verify_image_bytes` rejects it and it is never written.

**Writes go through a temp file and rename.** A crash mid-write cannot leave a
truncated file that looks valid.

**Computed geometry is preferred over raw EXIF.** `computed_geometry` and
`computed_compass_angle` are Mapillary's SfM-refined values and are materially
more accurate. Both are retained — their disagreement is a useful confidence
signal for Stage 3, so do not drop the raw values.

**Frames without a sequence id are discarded.** A lone frame has no baseline
partner and cannot contribute to Stage 3, so caching it wastes storage.

**The tiler snaps final edges to the parent bbox.** Repeated float addition
drifts downward (`0.018 + 0.009 == 0.026999999999999996`), which left an
unqueried sliver along the east and north edges of every study area. Found by
`test_final_edges_reach_the_parent_exactly`. Both regression tests stay.

## Next: Stage 1

Frame triage — blur (variance of Laplacian), exposure and glare, heading
filter, near-duplicate suppression. First OpenCV 5 code in the project, and the
first place COOL matters. Scheduled Aug 30.
