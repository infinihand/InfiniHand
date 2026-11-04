# InfiniHand: Streaming World-Space Hand Motion Estimation from Egocentric Video

[![Project Website](https://img.shields.io/badge/InfiniHand-Website-4CAF50?logo=googlechrome&logoColor=white)](https://infinihand.github.io/)
[![Model](https://img.shields.io/badge/Hugging_Face-Model-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/cskrren/InfiniHand)
[![arXiv](https://img.shields.io/badge/arXiv-2609.35743-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.35743)
[![Demo](https://img.shields.io/badge/Video-Demo-547354)](https://infinihand.github.io/#demo)

[Kerui Ren](https://scholar.google.com/citations?user=5kW5apkAAAAJ), Kaiwen Song, Weiguang Zhao, Yuxi Wang, Yufei Liu, [Bo Dai](https://daibo.info/), Haoyu Guo, Chunhua Shen, [Mulin Yu](https://scholar.google.com/citations?user=w0Od3hQAAAAJ), [Tao Lu](https://scholar.google.com/citations?user=Ch28NiIAAAAJ), Junting Dong

## Pipeline

InfiniHand estimates hand masks, camera-space MANO parameters and camera motion from egocentric RGB video. Streaming geometric features and hand-centered appearance are fused for hand reconstruction; camera poses transform the hands into a shared world frame. Optional sparse bundle adjustment refines the trajectory.

![InfiniHand pipeline](assets/method.png)

- **Stage I:** train mask and MANO reconstruction branches on short clips.
- **Stage II:** continue training the MANO and mask branches on longer clips and train the camera head. The camera head is trained only in Stage II.
- **Inference:** predict masks, hands and cameras, refine hands with default offline postprocessing, optionally refine cameras with sparse BA, and export camera-space and world-space geometry.

Training (`infinihand.train`) and inference (`infinihand.infer`) have separate entry points and share the model definitions.

## Environment setup

Requirements: Linux, Python 3.11, an NVIDIA GPU, and a CUDA-compatible PyTorch installation. CUDA extensions additionally require the CUDA toolkit and a C++ compiler.

```bash
git clone https://github.com/infinihand/InfiniHand.git
cd InfiniHand
conda create -n infinihand python=3.11 -y
conda activate infinihand
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For sparse BA, build the bundled DROID and LieTorch extensions:

```bash
python -m pip install --no-build-isolation ./third_party/droid
```

Obtain the licensed [MANO](https://mano.is.tue.mpg.de/) models and WiLoR's `mano_mean_params.npz`. Place them in `/path/to/weights/mano/`. Training expects `mano_data` under the configured WiLoR root to point to that directory. Follow the third-party terms in [NOTICE.md](NOTICE.md).

All external paths in the configurations are `/path/to/...` placeholders. Replace them before running. LingBot-Map and WiLoR runtime code is included in `infinihand/lingbot_map/` and `infinihand/wilor/`. Set the bundled configuration and asset paths for your checkout; set the MANO, data, checkpoint, cache and output locations to your installation.

## Inference

Download the inference checkpoint from [Hugging Face](https://huggingface.co/cskrren/InfiniHand):

```bash
mkdir -p /path/to/weights
curl -L https://huggingface.co/cskrren/InfiniHand/resolve/main/inifinihand_v1.pt \
  -o /path/to/weights/infinihand_v1.pt
```

### Inputs and intrinsics

Edit `configs/infinihand_v1.json`, including `mano_model_path`. The inference checkpoint contains the learned backbone and Mask/MANO/Camera weights; licensed MANO assets are loaded separately.

| Entry point | Image input | Intrinsics input | Processing |
|---|---|---|---|
| `python -m infinihand.infer` | Video file | Required `--intrinsics` JSON, one constant calibration for the video | Network + hand postprocessing + camera BA by default |
| `python -m infinihand.evaluation.camera_space` | Prepared dataset scenes | Per-frame intrinsics read by `UnifiedHandDataset` | Raw camera-space predictions and metrics; no hand postprocessing or BA |

The video CLI does **not** currently accept a dataset manifest or dataloader directly. Its intrinsics JSON must describe the original video resolution:

```json
{"fx": 700.0, "fy": 700.0, "cx": 640.0, "cy": 360.0, "image_width": 1280, "image_height": 720}
```

The CLI checks the decoded frame size, resizes RGB to 518×378, and scales `fx/cx` and `fy/cy` accordingly. Do not pre-scale the JSON to the network resolution. It reads frames in order without FPS resampling and processes at most `--max-frames` (default 8192); at least four frames are required. Use a fresh output directory for each run.

For dataset integration in Python, `run_dual_stream` accepts image tensors `[1,T,3,H,W]` and matching per-frame intrinsics `[1,T,3,3]` directly; it does not read JSON. The caller must supply RGB in `[0,1]` and intrinsics adjusted to the same image geometry. This lower-level function runs the network only; the video CLI separately orchestrates hand refinement, BA and export.

### Run video inference

Run from the repository root after installing the dependencies and replacing the asset paths. The following command uses the learned mask head, **fixed-beta → joint LSP depth (lambda 0.5) → SG5both** hand refinement, and guarded camera BA:

```bash
python -m infinihand.infer \
  --video /path/to/video.mp4 \
  --intrinsics /path/to/intrinsics.json \
  --config /path/to/InfiniHand/configs/infinihand_v1.json \
  --checkpoint /path/to/weights/infinihand_v1.pt \
  --mask-source base --mask-threshold 0.9 \
  --postprocess-lambda 0.5 \
  --droid-weights /path/to/weights/droid.pth \
  --output /path/to/outputs/demo
```

| Desired mode | Change to the command above |
|---|---|
| Hand refinement without camera BA | Add `--no-ba`; DROID weights and extensions are not needed |
| Network predictions only | Add both `--no-ba --no-postprocess` |
| Camera BA without hand refinement | Add `--no-postprocess` |
| Optional CUDA Graph acceleration | Add `--cuda-graphs` |
| SAM3 tracking instead of the learned mask head | Replace `--mask-source base` with the arguments below |

For SAM3, first follow [the setup guide](third_party/sam3_tracking/README.md), then supply all four external assets/environment arguments:

```bash
# Replace --mask-source base with these arguments:
--mask-source sam3-tracking \
--sam3-python /path/to/sam3-env/bin/python \
--sam3-source /path/to/sam3 \
--sam3-checkpoint /path/to/weights/sam3.pt \
--detector /path/to/weights/detector.pt
```

### Camera refinement (BA)

BA runs once after network inference, using every tenth **processed frame** plus the final frame. At 10 FPS the default spacing is one second. It does not optimize independently every ten frames, and its cost grows with the full sequence rather than a bounded sliding window.

| Option | Default | Meaning |
|---|---|---|
| `--ba-mode` | `fixed` | Fixed-stride keyframe selection |
| `--ba-stride` | `10` | Keyframe spacing in processed frames |
| `--ba-depth-init` | `none` | Initialize inverse depths to a constant, then optimize them |
| `--ba-depth-init sparse` | Optional | Initialize from RGB matches triangulated with predicted camera poses and supplied intrinsics |
| `--ba-no-reject-outliers` | Off | Disable abnormal-update rejection for ablations |

Both depth initialization modes use no ground truth and retain the predicted camera scale. The first pose anchors the world frame; corrections are interpolated to non-keyframes.

<details>
<summary>Abnormal-update rejection criteria</summary>

A keyframe reverts to its input pose if its translation correction exceeds `max(0.1 m, 5 × median correction)`, its rotation correction exceeds `max(10°, 5 × median correction)`, or its outgoing match weight is below 1% of the keyframe median in at least two of the first three BA iterations. Rejected poses are restored before dense interpolation. `BA_REPORT.json` records rejected frames and thresholds. These checks guard against large failures but do not guarantee an accuracy improvement on every scene.

</details>

### Outputs and timing

Hand slots are ordered **right, left**. World coordinates use the predicted camera scale.

| File | Contents | When written |
|---|---|---|
| `camera_space_raw.npz` | Original network hand predictions | Hand postprocessing enabled |
| `camera_space.npz` | Final camera-space MANO, joints and vertices | Always |
| `world_space.npz` | The same final hand geometry transformed by camera poses | Always |
| `POSTPROCESS.json` | Hand refinement settings, checks and timing | Hand postprocessing enabled |
| `trajectory.npz`, `BA_REPORT.json` | Refined camera trajectory and BA diagnostics | BA enabled |
| `SUCCESS.json` | Completion receipt, settings and timing | Successful completion |

The backbone overlaps with Camera/Mask/MANO heads on a second CUDA stream, preserving frame order and the 4-anchor + 16-recent-frame cache. Optional CUDA Graphs capture fixed-shape image encoders; they preserve precision and temporal computation but add startup cost and GPU memory use.

`SUCCESS.json` reports `feed_forward_fps` and `cuda_graph_setup_seconds`. Feed-forward FPS includes auxiliary prediction capture but **excludes** video decoding, file I/O, model loading, offline hand postprocessing and BA. It is not end-to-end application throughput.

## Hand postprocessing (enabled by default)

The inference CLI applies the following steps in order, using predictions only:

1. **Fix shape separately for each hand.** Estimate one beta vector for the right hand and one for the left over the input clip. Reject coefficient outliers with a median/MAD score (threshold 3.5), then average the retained estimates. Fewer than five samples use a median fallback. Re-decode MANO while preserving wrist position. This assumes one person per hand slot; split clips at identity changes.
2. **Refine depth with joint LSP.** Search camera-space wrist depth from 0.10 to 1.20 m in 5 mm steps. For every candidate, recompute both lateral translation components with LSP using the predicted 2D joints and the actual crop geometry. Score the projected MANO silhouette against the predicted hand mask with `I / (I + lambda * FP + FN)`, where `FP` is MANO outside the mask and `FN` is uncovered mask area. **The default lambda is 0.5.** A temporal dynamic program penalizes depth changes by `0.02 * (delta_z / 0.01 m)^2 * (0.1 s / delta_t)^2`. Empty masks or infeasible depth grids retain the fixed-shape original wrist before smoothing; counts and boundary selections are reported.
3. **Apply SG5both.** Smooth wrist `tx/ty/tz` and the global plus 15 local MANO rotations separately with a five-frame quadratic fit. Rotation filtering uses local SO(3) tangents, not Euler-angle averaging. Beta remains fixed. Missing detections and gaps greater than 1.5 nominal frame intervals split the filter; missing hands are not filled in. Fewer than three consecutive valid frames remain unsmoothed. At regular 10 FPS this is the evaluated SG5 filter; actual timestamps are used for irregular spacing.

The learned mask head supplies the refinement masks by default. With `--mask-source sam3-tracking`, refinement uses the resized binary SAM3 masks. No ground-truth masks, joints, depth or poses are used. The mask-guided search requires CUDA and Triton (included with the documented Linux CUDA PyTorch installation). Licensed MANO assets must match the model's geometry; reconstruction and LSP convention checks fail explicitly on mismatch.

```bash
# Append to the inference command to change the mask penalty:
--postprocess-lambda 0.5

# Or disable all three hand refinement steps:
--no-postprocess
```

Postprocessing is **offline**, even though network inference is streaming: beta estimation and the depth path use the whole clip, and centered SG5 uses future frames (normally two, up to four at the boundaries). The five-frame window follows the input video FPS: at 10 FPS its normal lookahead is 0.2 s; this command does not resample videos to 10 FPS. The CLI derives frame times from the decoded video's declared constant FPS. For externally timestamped sequences, call `refine_hands` with the original timestamps. Fast motion and inaccurate masks can reduce alignment quality; the depth range may be unsuitable for distant hands.

`camera_space_raw.npz` retains the network prediction. `camera_space.npz` contains the refined MANO parameters and matching joints/vertices; `world_space.npz` transforms that same refined geometry with the predicted/BA-refined camera poses. Camera pose estimation and BA settings are unchanged. `POSTPROCESS.json` records beta estimation, lambda, optimized/fallback/boundary hand counts, convention checks and postprocessing time. `SUCCESS.json` also contains that report. `feed_forward_fps` measures only the forward pipeline (including auxiliary capture), **not** offline refinement or BA; see the separate timing fields for total application cost. When disabled, no raw duplicate or postprocessing report is written and the existing camera-space output is preserved.

## Training

Prepare unified scene manifests and set `dataset_jsons` and `sampling_weights` in `configs/training/`. All training manifests must declare `split: train`. See [the data format](data/README.md) for the manifest and geometry requirements. Place official LingBot-Map and WiLoR initialization weights at the configured paths.

### Training Stage I

Stage I trains the MANO and mask branches on four-frame clips. Train the MANO branch, then initialize the mask branch from the same complete base written by that run:

```bash
python -m infinihand.train --config /path/to/InfiniHand/configs/training/stage1_mano.yaml \
  --set output_dir=/path/to/runs/stage1_mano
python -m infinihand.train --config /path/to/InfiniHand/configs/training/stage1_mask.yaml \
  --set finetune_checkpoint=/path/to/runs/stage1_mano/initial_model.pt \
  --set output_dir=/path/to/runs/stage1_mask
```

### Training Stage II

Stage II uses 36-frame clips to continue training the MANO and mask branches and to train the camera head. MANO and mask each have Stage I and Stage II configurations; the camera head has only a Stage II configuration. Set `finetune_checkpoint` to a complete Stage I checkpoint containing the trained MANO and mask weights:

```bash
for task in mano mask camera; do
  python -m infinihand.train \
    --config /path/to/InfiniHand/configs/training/stage2_${task}.yaml \
    --set finetune_checkpoint=/path/to/weights/stage1.pt \
    --set output_dir=/path/to/runs/stage2_${task}
done
```

Batch size is per process; global batch size follows the number of processes. For distributed training, replace `python -m infinihand.train` with `torchrun --standalone --nproc_per_node=8 -m infinihand.train`. Use `--check-config` to validate training manifests and initialization paths before launching. Resume by supplying the run's complete base as `finetune_checkpoint` and its partial update as `resume_checkpoint`; the trainer restores optimizer and scheduler state.

## Evaluation

Edit `configs/evaluation/camera_space.yaml` to set the single inference checkpoint, model configuration, and test manifests for ARCTIC, HOT3D, EgoDex and HOI4D. Evaluation reads images and per-frame intrinsics through `UnifiedHandDataset`, uses predicted mask boxes, and reports raw camera-space joint errors and coverage for each scene. It does not require a separate intrinsics JSON and does not currently apply the video CLI's hand postprocessing or BA.

```bash
python -m infinihand.evaluation.camera_space \
  --config /path/to/InfiniHand/configs/evaluation/camera_space.yaml
```

Use `--dataset arctic` to select one dataset, `--scene SCENE_NAME` for one scene, or `--check-config` to validate the configuration before loading the model.

## Acknowledgement

We thank the authors of [LingBot-Map](https://github.com/Robbyant/lingbot-map), [WiLoR](https://github.com/rolpotamias/WiLoR), [HaWoR](https://github.com/ThunderVVV/HaWoR), and [ViDiHand](https://github.com/NTUYWANG103/ViDiHand) for their excellent work and contributions to the research community.

## Contact and citation

For questions, contact [Kerui Ren](mailto:renkerui@sjtu.edu.cn). Please cite our paper:

```bibtex
@article{ren2026infinihand,
  title={InfiniHand: Streaming World-Space Hand Motion Estimation from Egocentric Video},
  author={Ren, Kerui and Song, Kaiwen and Zhao, Weiguang and Wang, Yuxi and Liu, Yufei and Dai, Bo and Guo, Haoyu and Shen, Chunhua and Yu, Mulin and Lu, Tao and others},
  journal={arXiv preprint arXiv:2609.35743},
  year={2026}
}
```
