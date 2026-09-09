# Licensing, attribution and privacy record

Judges score cloud delivery, reproducibility and **responsible operation** at
10%. This file is where the licensing chain is made auditable. Keep it current
as sources are added; an out-of-date version is worse than none.

## Imagery — Mapillary

- **Licence:** CC BY-SA 4.0. Every image on Mapillary falls under an open data
  licence; contributors retain rights to their images while permitting others
  to view and distribute with proper attribution.
- **Share-alike:** derived products that carry image content inherit
  share-alike obligations. Rendering source frames in the UI is carrying image
  content. Publishing only derived *measurements* (slope, curb height) is a
  weaker case, but the safe course is to attribute in both.
- **Required attribution:** integrating data extracted through the Mapillary
  API or vector tiles requires visibly displaying the Mapillary logo with a
  link back to mapillary.com.
- **Where it is discharged in this project:**
  - `manifest.json` records provenance for every ingested area
    (`source.license`, `source.attribution_note`); asserted by
    `tests/test_ingest.py::TestIngestRun::test_manifest_carries_licensing_provenance`
  - Map UI footer — Mapillary logo, link, and CC BY-SA notice **(Stage 5, not
    yet implemented — do not ship the UI without it)**
  - Technical report, responsible-use section

## Privacy

- Mapillary applies automated blurring to faces and licence plates before
  publication. This project does not undo, reconstruct or circumvent that
  blurring, and must not.
- This pipeline stores no personally identifiable information. Cached artefacts
  are image bytes, geographic coordinates, capture timestamps and camera
  metadata.
- **Own-captured imagery:** anything the team captures and uploads goes through
  Mapillary's blurring pipeline on upload. Do not add raw team captures to the
  cache by any route that bypasses it.
- Findings are attached to *street corners and segments*, never to addresses as
  subjects. The reachability output names addresses as origins of a routing
  query. It does not describe residents, and the UI must not present it as if
  it does.

## Training data — Mapillary Vistas (Stage 2)

- 25,000 street-level images with pixel-wise annotation, offered free to
  academic and commercial researchers; commercial *product integration*
  requires separate licensing.
- **Action before Stage 2 begins:** read the current Vistas terms in full and
  record here whether they cover a competition entry. Do not start training
  against an unresolved licence.

## Street network — OpenStreetMap (Stage 4)

- Open Database License (ODbL). Attribution required. Derived databases carry
  share-alike obligations.
- Accessed via `osmnx`; attribution goes in the UI footer alongside Mapillary.

## Software

| Component | Licence | Note |
|---|---|---|
| OpenCV 5 | Apache 2.0 | Core dependency from Stage 1 |
| COOL (Cloud-Optimized OpenCV Library) | Per AWS Marketplace terms | **Record the exact terms and version here once provisioned** |
| requests, boto3, Pillow, NumPy | Apache 2.0 / MIT / BSD-3 / BSD-3 | See `requirements.txt` |

## Outstanding items

- [ ] Confirm Vistas terms cover this use — before Stage 2 training begins
- [ ] Record COOL licence terms and version — at Graviton deployment (week 3)
- [ ] Attribution rendering in the map UI — Stage 5, blocking on UI ship
- [ ] Decide and document the licence for this project's own released code