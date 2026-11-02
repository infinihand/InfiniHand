# Source map

This repository extracts the detection + SAM3 mask path from HaWoR commit
`579253db1787b306161e81444170178cd6628e37`.

| Standalone file | Production source |
|---|---|
| `detection.py` | `pipeline.py::save_yolo_detector_area_max_bboxes` |
| `detection_payload.py` | `detection_payload.py` |
| `sam3_point.py` | `sam3_point.py` |
| `sam3_helpers.py` | the helper surface used from `sam_anchor.py` and `sam_repropagate.py` |
| `ego_candidate_score.py` | `ego_candidate_score.py` |

Configuration values are documented in `PRODUCTION_DEFAULTS.md` and `configs/default.yaml`. SAM3 source and weights are provided by the user.

Standalone-only glue does three things that the larger HaWoR pipeline used to
do around those functions:

1. decode/normalize input frames;
2. resize frames and boxes together for the SAM3 1008-pixel input boundary;
3. restore binary masks to original decoded-frame coordinates before writing
   the handoff masks and QA overlay.

The CUDA-invalid automatic retry from the production worker is intentionally
not carried into the handoff CLI. A runtime failure stops at the failing call.
