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
- **Inference:** predict masks, hands and cameras, optionally refine cameras with sparse BA, and export camera-space and world-space geometry.

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

Edit `configs/infinihand_v1.json`, including `mano_model_path`. The checkpoint contains the learned backbone and Mask/MANO/Camera weights; licensed MANO assets are loaded separately. Prepare intrinsics for the original video resolution:

```json
{"fx": 700.0, "fy": 700.0, "cx": 640.0, "cy": 360.0, "image_width": 1280, "image_height": 720}
```

Run inference with the learned mask head and guarded BA:

```bash
python -m infinihand.infer \
  --video /path/to/video.mp4 --intrinsics /path/to/intrinsics.json \
  --config /path/to/InfiniHand/configs/infinihand_v1.json \
  --checkpoint /path/to/weights/infinihand_v1.pt \
  --mask-source base --droid-weights /path/to/weights/droid.pth \
  --output /path/to/outputs/demo
```

Use `--no-ba` for feed-forward-only inference. The BA defaults are `--ba-mode fixed --ba-stride 10 --ba-depth-init none`, with abnormal-update rejection enabled. BA selects every tenth **processed frame**, includes the last frame, and runs once over these keyframes after feed-forward inference; it does not run a separate optimization every ten frames. At 10 FPS, the keyframe spacing is one second. Keyframe count and full-sequence BA cost grow with video length; this implementation does not provide bounded-cost sliding-window optimization.

Use `--ba-depth-init sparse` to initialize inverse depth from RGB feature matches triangulated with Camera head poses and the supplied intrinsics. This uses no ground-truth poses or depth, inherits the predicted camera scale, and leaves depth free to change during BA. The default `none` uses constant inverse-depth initialization, also optimized by BA.

Update rejection uses no ground truth. A keyframe falls back to its input camera pose if its translation correction exceeds `max(0.1 m, 5 × median correction)`, its rotation correction exceeds `max(10°, 5 × median correction)`, or its outgoing match weight is below 1% of the keyframe median in at least two of the first three BA iterations. The first pose anchors the world frame. Dense pose corrections are interpolated after rejected keyframes are restored. This guards against large failures but does not guarantee improved accuracy on every scene. `--ba-no-reject-outliers` disables it for ablations. `BA_REPORT.json` records rejected frames and thresholds.

Optional SAM3 tracking uses `--mask-source sam3-tracking` and the assets described in [the SAM3 setup guide](third_party/sam3_tracking/README.md).

Outputs include `camera_space.npz`, `world_space.npz` and `SUCCESS.json`; BA also writes `trajectory.npz`. Hand slots are right, left. Use a fresh output directory. Each invocation accepts 4–8192 frames; BA runs after feed-forward prediction. World coordinates use the predicted camera scale.

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

Edit `configs/evaluation/camera_space.yaml` to set the single inference checkpoint, model configuration, and test manifests for ARCTIC, HOT3D, EgoDex and HOI4D. Evaluation uses predicted mask boxes and reports camera-space joint errors and coverage for each scene.

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
