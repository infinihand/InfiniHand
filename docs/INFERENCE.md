# InfiniHand inference

Inference-only extraction: **Learned masks -> InfiniHand MANO and camera prediction -> sparse DROID BA -> world-space hands**.

The default built-in mask head uses `--mask-source head --mask-threshold 0.9`. Both paths use the complete checkpoint explicitly supplied through `--checkpoint`.

## Weights

Export a plain tensor state dictionary containing all learned model weights from the current training runs using `python -m infinihand.export_weights`. Its geometry convention must be wrist depth. Architecture settings live separately in `configs/infinihand_v1.json`. The public export omits MANO asset buffers, which are loaded from your licensed local assets. No checkpoint is selected implicitly.

- Place licensed MANO assets and `mano_mean_params.npz` in `/path/to/weights/mano/`, following WiLoR's `mano_data` layout. MANO terms apply separately.
- Optional SAM3 tracking additionally needs a detector checkpoint, the gated SAM3 checkpoint and installed SAM3 source. See `third_party/ego_step2/README.md`.
- Sparse BA needs `/path/to/weights/droid.pth` from HaWoR/DROID-SLAM.

## Environment

Linux, CUDA and a compatible PyTorch installation are required. Install PyTorch first for your GPU, then:

```bash
pip install -r requirements.txt
# Builds the vendored DROID and LieTorch CUDA extensions:
(cd third_party/droid && pip install --no-build-isolation .)
```

SAM3 may use its **own Python environment**; install SAM3 and the requirements from `third_party/ego_step2/requirements.txt` there. Pass `--sam3-python` and `--sam3-source` to select it. BA can likewise use a compatible CUDA-extension environment through `--ba-python`.

## Run

Provide original-image intrinsics, e.g. `camera.json`:

```json
{"fx": 700.0, "fy": 700.0, "cx": 640.0, "cy": 360.0, "image_width": 1280, "image_height": 720}
```

```bash
python -m infinihand.cli \
  --video /path/to/input.mp4 --intrinsics /path/to/camera.json --output /path/to/outputs/example \
  --checkpoint /path/to/current_model.pt \
  --mask-source sam3 --sam3-source /path/to/sam3 --sam3-checkpoint /path/to/weights/sam3.pt \
  --detector /path/to/weights/detector.pt --droid-weights /path/to/weights/droid.pth
```

Optional internal mask head:

```bash
python -m infinihand.cli --video /path/to/input.mp4 --intrinsics /path/to/camera.json \
  --output /path/to/outputs/head --checkpoint /path/to/current_model.pt --mask-source head --mask-threshold 0.9
```

`--no-ba` exports the camera-head trajectory without refinement. `--ba-stride` defaults to **20**. Use `--ba-mode binary` (default) for pool-index selection or `--ba-mode fixed` for fixed frame selection. Use a fresh output directory on each run. Intrinsics must match the decoded image dimensions; the code does not silently guess focal length or correct mismatched calibration.

## Outputs and conventions

- `camera_space.npz`: joints, vertices, global orientation, MANO pose/shape, translation, validity and camera encoding.
- `trajectory.npz`: BA-refined camera-to-first-camera trajectory, initial trajectory and selected keyframes (when BA enabled).
- `world_space.npz`: transformed geometry/MANO global orientation and translation, plus unchanged local pose/shape.
- `SUCCESS.json`: completed-frame count, selected mask path, timing and memory-contract receipt.
- SAM3 outputs preserve the original per-side masks and metadata.

Hand slots are **right, left**. Pose/orientation matrices use the saved model's conventions. Camera-to-world transforms map column vectors into the first predicted camera frame. World outputs retain the camera head's learned translation scale; this wrapper performs no GT alignment or independent metric-scale recovery. Monocular scale accuracy is not guaranteed.

The streaming backbone retains four anchor frames and sixteen recent dense frames. This wrapper stages decoded frames in memory and currently supports at most 8,192 frames per invocation; it is not an unbounded video loader. BA operates after prediction with binary-pool or fixed-stride selection, hand-box exclusion, gauge normalization tracking and interpolation of pose corrections. Execution is sequential, not asynchronous. No throughput claim is made for this wrapper.

## Provenance

- EgoPreprocess step2: `cskrren/EgoPreprocess`, commit `2e33f7647d3f32248c207e601b1b6adb534684ae`; point seeds + joint propagation, not independent per-frame segmentation.
- InfiniHand: dual-stream camera and hand inference runtime.
- Sparse BA: DROID camera refinement with binary-pool or fixed-stride selection.

Third-party code retains its upstream licenses/notices. See `NOTICE.md`. Checkpoints and MANO assets have their own terms.
