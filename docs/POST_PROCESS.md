# Hand postprocessing

Both video and dataset inference enable this processing by default (lambda = 0.5). See [inference commands](../README.md#inference).

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

Postprocessing is **offline**, even though network inference is streaming: beta estimation and the depth path use the whole clip, and centered SG5 uses future frames (normally two, up to four at the boundaries). The five-frame window follows the input video FPS: at 10 FPS its normal lookahead is 0.2 s; this command does not resample videos to 10 FPS. Video inference derives frame times from the decoded video's declared constant FPS. Dataset inference preserves increasing source timestamps; when timestamps are missing, it requires an explicit `--dataset-fps` fallback describing the processed frames. Fast motion and inaccurate masks can reduce alignment quality; the depth range may be unsuitable for distant hands.

`camera_space_raw.npz` retains the network prediction. `camera_space.npz` contains the refined MANO parameters and matching joints/vertices; `world_space.npz` transforms that same refined geometry with the predicted/BA-refined camera poses. Camera pose estimation and BA settings are unchanged. `POSTPROCESS.json` records beta estimation, lambda, optimized/fallback/boundary hand counts, convention checks and postprocessing time. `SUCCESS.json` also contains that report. `feed_forward_fps` measures only the forward pipeline (including auxiliary capture), **not** offline refinement or BA; see the separate timing fields for total application cost. When disabled, no raw duplicate or postprocessing report is written and the existing camera-space output is preserved.

