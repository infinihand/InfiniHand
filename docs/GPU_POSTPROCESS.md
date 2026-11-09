# Packed-mask GPU hand refinement

This is the default implementation behind `infinihand.postprocess.refine_hands`.
It accelerates the [existing hand-refinement algorithm](POST_PROCESS.md) without
changing its depth candidates, mask score, temporal penalty or smoothing rule.
It refines predictions; it does not use ground truth or replace network inference.

## Code map

| File / entry | Responsibility |
|---|---|
| `infinihand/postprocess.py::refine_hands` | Public API, validation, dispatch to the packed GPU backend |
| `infinihand/_postprocess_gpu.py::refine_hands_gpu` | Batched shape decoding, depth search, temporal DP, SG5, final decoding and audit |
| `infinihand/_postprocess_packed.py` | Triton silhouette rasterization, target packing, population-count scoring |
| `infinihand/postprocess.py::refine_hands_reference` | Original implementation retained for numerical regression comparisons |
| `infinihand/stabilize_shape.py` | Reference MAD beta estimation and shape decoding |
| `tests/test_hand_postprocess.py` | Reference temporal, validity and shape contracts |
| `tests/test_packed_masks.py` | SO(3), packed raster, image-boundary, empty-mask and scoring contracts |

The underscore-prefixed modules are implementation details. Call the public API
rather than a Triton kernel directly. No checkpoints or MANO assets are bundled
with this implementation; use the matching licensed MANO assets.

## Processing stages and precision

1. Upload the clip and initialize right/left MANO models.
2. Estimate fixed beta separately per hand using median/MAD filtering. Re-decode
   the original beta to check asset compatibility, then decode the fixed shape.
3. Batch valid frames and search 221 camera-space wrist depths from 0.10 to
   1.20 metres in 0.005-metre increments. Each depth uses the unchanged 25-step
   crop-space bisection and LSP translation solve. Flatten `(frame, depth)` into
   candidate batches for projection, packed rasterization and mask scoring.
4. Select the depth path with the existing temporal dynamic program. A small
   availability readback determines segment boundaries; score arrays and
   backtracking stay on GPU.
5. Apply SG5 to wrist translations and rotation tangents, independently within
   valid segments. CPU builds window indices; polynomial fitting and SO(3)
   arithmetic run on CUDA.
6. Decode the final joints/vertices from the smoothed pose and fixed shape.
   Transfer results and compact audit values back to the host.

| Operation | Arithmetic |
|---|---|
| Beta statistics, MANO decode, SG5 | FP64, matching the reference |
| LSP, depth search and projection | FP32 |
| Raster coverage and mask counts | Integer/bit operations, with FP32 edge tests |
| Mask scores and DP costs | FP64; the depth penalty retains the reference FP32 construction |

Autocast is disabled inside GPU refinement. Returned prediction arrays retain
the input array dtypes. This is not an all-FP32 rewrite of the algorithm.
CPU asset loading, checks, scheduling and report packaging remain. There are no
per-frame candidate-score transfers to CPU, but the routine is not completely
host-free or synchronization-free.

## Exact mask representation

Let `R=max(H,W)`. Original pixel coordinates are kept: images are padded, not
resized. Pixel `(x,y)` maps to:

```text
p = y * R + x
word_index = p // 32
bit_index  = p % 32
```

A word holds 32 binary pixels. Words are stored as `int32` and interpreted as
`uint32` inside kernels. Final-word padding bits are zero. Triangle coverage uses
the same edge tests as the reference; atomic OR combines overlapping triangles
into a silhouette. This is silhouette coverage, not a depth-buffer rendering.
Raster writes are clipped to the real `H×W` image rectangle.

The target mask is packed once per frame batch. For each candidate:

```text
I  = popcount(rendered & target)
A  = popcount(rendered)
FP = A - I
FN = target_area - I
score = I / max(I + lambda_fp * FP + FN, 1)
```

Counts are exact integers; the division uses FP64. A packed raster buffer needs
`4*ceil(R*R/32)` bytes per candidate: approximately 32× less than the original
int32 raster buffer, or 8× less than a dense bool mask. This does not imply the
same reduction for total process VRAM, which includes geometry and other tensors.

## Public API and coordinate contract

```python
from infinihand.postprocess import refine_hands

refined, audit = refine_hands(
    prediction,                 # NumPy arrays captured from the same forward pass
    intrinsics,                 # [T, 3, 3], matching the mask image coordinates
    masks,                      # bool [T, 2, H, W]
    timestamps,                 # [T], finite, strictly increasing seconds
    mano_model_path,             # matching licensed MANO model directory
    lambda_fp=0.5,
    device="cuda",
    frame_batch=1024,
    raster_batch=4096,
)
```

Slots are **right, left**, not image-left/image-right. Camera-space coordinates
and wrist translations are in metres. `T` must be positive. Required prediction
fields are:

| Field | Shape | Meaning |
|---|---|---|
| `valid` | `[T,2]` | Input hand validity; preserved |
| `betas` | `[T,2,10]` | Shape coefficients for matching MANO assets |
| `camera_global_orient` | `[T,2,3,3]` | Camera-space global rotation matrices |
| `hand_pose` | `[T,2,15,3,3]` | Local rotation matrices, not axis angles |
| `camera_translation` | `[T,2,3]` | Camera-space wrist position |
| `camera_joints` | `[T,2,21,3]` | Decoded joints; joint 0 is the wrist |
| `camera_vertices` | `[T,2,778,3]` | Consistent decoded camera-space mesh |
| `pred_keypoints_2d` | `[T,2,21,2]` | Native centered crop-normalized predictions; code adds 0.5 before LSP |
| `mano_virtual_rotations` | `[T,2,3,3]` | Camera-to-virtual-crop rotation convention from the forward pass |
| `mano_crop_intrinsics` | `[T,2,3,3]` | Intrinsics for the native `(H,W)=(256,192)` hand crop |
| `lsp_valid_mask` | `[T,2,21]` | Joint participation in LSP |

For row vectors, camera-relative joints map to crop space as `j_camera @ R.T`,
and crop-space translations map back as `t_crop @ R`. Do not apply another
left-hand flip or substitute image-space joints for `pred_keypoints_2d`.
The auxiliary fields must come from the same forward pass as the geometry.
The exported `camera_space_raw.npz` alone does **not** contain all these auxiliary
fields; use the CLI or explicitly capture them before export to replay refinement.

The API returns a new prediction dictionary plus a JSON-serializable audit.
No `world_*` inputs are accepted. Convention checks reject original MANO/LSP
reconstruction errors exceeding `2e-5` metres. Validity is not filled in. Empty
masks/infeasible depths retain the fixed-shape wrist before smoothing; valid
frames can still move through SG5. See the algorithm document for gap handling
and identity assumptions.

## CLI and batch tuning

Append these options to either documented [inference command](../README.md#inference):

```bash
# Defaults: maximum concurrent frames for LSP, and raster candidates
--postprocess-frame-batch 1024 --postprocess-raster-batch 4096

# Lower temporary memory
--postprocess-frame-batch 128 --postprocess-raster-batch 1024

# Unchanged score control and disable switch
--postprocess-lambda 0.5
--no-postprocess
```

`frame_batch` batches valid frame/hand observations **within one hand slot**.
`raster_batch` batches flattened frame-depth candidates; it is not the number
of depth bins per frame. Smaller values are useful alongside a resident network.
They do not alter the candidate grid or mix different clips into one temporal
sequence. The entire input clip and its results still reside in memory: these
options bound temporary work, not total memory for arbitrarily long videos.

More allocated memory is not necessarily faster. The uncompressed implementation
mostly saturated on raster/score traffic; changing its representation produced
a much larger gain than merely increasing batch sizes.

## Audit and timing fields

| Report field | Interpretation |
|---|---|
| `backend`, `frame_batch`, `raster_batch` | Implementation and limits actually used |
| `shape` | Per-hand fixed beta and median/MAD filtering statistics |
| `input_valid_hands`, `optimized_hands`, `fallback_hands`, `boundary_hands` | Coverage and depth-grid diagnostics |
| `selected` | Per-frame/hand depth indices; `-1` means no selected candidate |
| `max_decode_error_m`, `max_lsp_reproduction_error_m` | Input geometry/convention checks |
| `seconds` | Synchronized refinement wall time, including transfer/setup/checks |
| `stage_ms` | CUDA-event intervals for upload, setup, decode, search/score, DP, SG5 and download |
| `lsp_ms`, `raster_score_ms` | Summed event intervals inside the candidate batches |
| `peak_allocated_gib` | Process CUDA allocator high-water mark, not necessarily refinement-only VRAM |

CUDA-event intervals can contain CPU launch gaps; they are not kernel-only
profiler results. Peak memory includes resident models and earlier work unless
the caller resets the allocator peak before the measurement. `seconds` excludes
network inference, input-video decoding, dataset I/O, output-file writing and
visualization. First-ever Triton compilation and cold initialization add latency.
None of these numbers is end-to-end inference FPS.

## Measured regression and performance

A800-SXM4-80GB, ARCTIC `s05_box_grab_01_0`, 675 frames and 1,163 valid hand
observations. All settings reused the same cached prediction/mask/auxiliary
arrays from the new EgoBench checkpoint with SHA-256:

```text
acc2ec7cc9136b2ff5e7ab260c4708585e92e87ec34881f38c38074ab3ca7859
```

| Implementation | Frame/candidate batch | Seconds | Standalone peak allocated GiB |
|---|---:|---:|---:|
| Original reference | per-frame/16 | 31.42 | not recorded |
| Dense GPU batching | 64/512 | 3.70 | 1.61 |
| Dense GPU batching | 675/4096 | 3.35 | 10.79 |
| Packed GPU batching | 128/1024 | 0.97 | 0.61 |
| Packed GPU batching | 675/4096 | 0.830–0.833 | 1.69 |

The last row is three warm repeats. The first call in that process took 1.55
seconds. The packaged public API was separately checked with its default
1024/4096 limits and took 1.68 seconds on its first call; the effective frame
batch contains only available observations, so the same 675-frame clip fits.
These are single-scene measurements, not a multi-dataset speed guarantee.

All full-scene selected depth paths matched the reference. Maximum valid
joint/vertex coordinate difference was `5.96e-8` metres. Sixteen sampled hand
observations had zero differing raster pixels, exact target packing and identical
scores. This establishes numerical agreement on the tested data, not improved
reconstruction accuracy: the unchanged refinement can still worsen absolute
alignment when masks or geometry are inaccurate.

## Running the tests

From the repository root in the documented inference environment:

```bash
python -m unittest discover -s tests -p 'test_hand_postprocess.py' -v
python -m unittest discover -s tests -p 'test_packed_masks.py' -v
```

The reference and SO(3) tests do not require licensed assets. The packed-mask
test requires CUDA and Triton, and is skipped without CUDA. It checks both image
orientations, triangles crossing image boundaries, degenerate geometry, padding
bits, empty masks and several score penalties. A skipped CUDA test does not
validate the raster kernels. Full-scene regression additionally requires matching
MANO assets and the original same-forward-pass auxiliary arrays.
