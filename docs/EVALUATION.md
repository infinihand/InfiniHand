# Camera-space evaluation

Supply held-out scene manifests separately from training. `configs/evaluation/camera_space.yaml` specifies model and geometry settings. `configs/evaluation/checkpoints.json` specifies a complete Stage I base, a Stage II update and a mask update. Set those paths to compatible exports from your runs; no checkpoint is selected automatically.

Inspect the supported stages and input schema with:

```bash
python -m infinihand.evaluation.camera_space --help
python -m infinihand.evaluation.report --help
```

Evaluation exports source frame IDs, calibrated intrinsics and target validity alongside predictions. Runtime hand slots are right, left. Ground-truth boxes are an auxiliary control; distinguish them from SAM3 and learned-mask box modes in reports. Keep scene selection, frame limits, scoring and checkpoint provenance fixed when comparing models. Preserve the generated input and result records with the experiment outputs.
