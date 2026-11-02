# Sparse binary-pool BA inference

Entry point: `python -m infinihand.infer`. Use `--ba-mode fixed` for fixed-stride BA.

This implements Appendix C (Eq. 34–35) of https://arxiv.org/abs/2609.35743:

1. Append each frame in chronological order with bit 0.
2. Trigger when pool length is **greater than** N, not greater than or equal.
3. Select pool entries `0, K, 2K, ...`, starting at the oldest entry. K is a
   **pool-index** stride, not a stride in original frame IDs.
4. Run the DROID backend on those frames, initialized by current predicted c2w.
5. Only on successful finite output, toggle the selected bits. Remove all zero
   entries, including unselected new frames; retain unselected old bit-1 frames.
   An old frame selected for retirement still contributes to the current BA call.

N=160, K=20 are configurable implementation defaults, not paper-reported settings.
An optional end-of-video flush uses the same selection/retention rules even below
capacity; `--ba-no-flush` disables this extension. A selection with fewer than two
frames performs no optimization and is reported as zero BA calls.

## Camera checkpoint

Supply an exported camera-head checkpoint at `/path/to/weights/camera.pt`.

`--checkpoint` still requires a complete exported inference state dict (backbone,
MANO, mask, etc.). `--camera-checkpoint` overlays **only**
`backbone.model.camera_head.*`, requiring an exact head match and finite tensors.
It rejects mixed/full checkpoints rather than accidentally changing hand weights.
It reads trusted PyTorch training checkpoints; optimizer state is not applied.
The selected checkpoint path, SHA256 and training step are recorded in SUCCESS.json.

Example, replace the video, calibration, and complete inference weight paths:

```bash
cd /path/to/infinihand
/path/to/ba-python -m infinihand.infer \
  --video /path/to/video.mp4 --intrinsics /path/to/intrinsics.json \
  --checkpoint /path/to/complete_inference_weights.pt \
  --camera-checkpoint /path/to/weights/camera.pt \
  --mask-source head \
  --ba-mode binary --ba-pool-capacity 160 --ba-stride 20 \
  --ba-python /path/to/ba-python \
  --droid-weights /path/to/weights/droid_verified.pth \
  --output /path/to/fresh_output
```

The learned mask head is the default; SAM3 tracking is available with `--mask-source sam3` and its asset arguments.
Use `--ba-mode fixed` for a fixed-original-frame-stride control, or `--no-ba` for
feed-forward results only. DROID requires its CUDA extensions; the camera inference
Python environment alone does not include them, hence the separate `--ba-python`.

## Coordinate and execution conventions

Input/output trajectories are OpenCV camera-to-world poses. DROID receives inverse
w2c poses and returns c2w. Each call is rigidly anchored to the first selected
input camera, preserving the existing world gauge rather than resetting the world.
DROID's numerical disparity normalization scale is explicitly undone. This does
**not** recover a physical metric scale or guarantee removal of monocular scale drift.

For non-keyframes, we interpolate world-frame left corrections with rotation SLERP
and linear translation, holding endpoint corrections outside the selected interval.
Future incoming frames inherit the latest correction. Corrected camera poses then
transform camera-space hands to world-space hands; the hand predictor is unchanged.
This interpolation is an implementation choice beyond the paper's pool equations.

The current CLI executes the pool incrementally in temporal order **after** the
feed-forward pass, in a separate sequential subprocess. It is not concurrent BA.
The pool bounds the keyframes used for each optimization; full exported trajectories
and the existing inference RGB tensor are still retained for the offline video.
Each BA call re-encodes its selected frames; persistent GPU feature/depth caching and
asynchronous scheduling are future performance work, not claimed here.

## Outputs and validation

- `camera_space.npz`: unchanged feed-forward hand predictions.
- `ba_inputs.npz`: initial trajectory and predicted bounding boxes.
- `trajectory.npz`: initial/refined c2w and left corrections.
- `BA_REPORT.json`: selected IDs, pool before/after, flags, BA diagnostics and timing.
- `BA_SUCCESS.json`: completion and actual BA call count.
- `world_space.npz`: hands transformed by the refined trajectory.

`tests/test_sparse_ba.py` checks selection/toggle/retirement, failure without eviction,
long-sequence identity behavior, nonidentity world gauge, interpolation, invalid
inputs, and short-video handling.

The backend also supports the installed indexed 6-argument DROID alt-correlation
CUDA ABI via `droid_corr_compat.py`, while retaining the bundled 4-argument path.
This adapter is inference-only, keeps correlation scaling unchanged, and does not
modify vendor files. The Camera checkpoint's `_mano_translation_convention` marker
is explicitly ignored; it is never overlaid onto the hand model.
