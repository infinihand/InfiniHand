# Sample filtering and random valid-hand training

Applies to InfiniHand v1's four configs in `configs/training/`.

## Loss acceptance

- The first 1,000 attempted training steps use an inclusive loss threshold of 10.0.
- Later steps use the mean of the previous 1,000 attempted steps' finite raw loss means, aggregated across ranks. The current step is appended only after its decision. Nonfinite observations are excluded; a wholly nonfinite step occupies a window position without adding a zero.
- Finite loss values rejected by the threshold remain in history. Keeping only accepted values would progressively lower the threshold.
- The filtering unit is one batch sample (one sequence/window). Its complete weighted objective includes every enabled mask, camera-space, camera-pose, world-joint, and temporal term. Each sample's terms retain their own valid-element normalization; accepted samples receive equal weight.
- A rejected sample does not discard its batch peers or other ranks. Its graph is discarded and accepted inputs are replayed. A finite loss with nonfinite gradients triggers isolated per-sample replay; only finite gradients are accumulated.
- DDP automatic reduction is disabled during forward/backward. Accepted loss sums are explicitly reduced in buckets and normalized by the global accepted-sample count. Globally unused parameters keep `grad=None`; a globally empty step does not call the optimizer.
- Checkpoints save `loss_filter_state` with protocol `finite_raw_step_mean_v1`, including empty-step positions. Resume uses this state instead of old accepted-only histories.

## MANO training selection

Stage I MANO and Stage II joint training select at most one valid hand per frame. If only one side is valid it is selected; if both sides are valid each has probability 1/2. Frames with no valid crop run no hand branch. This is actual compact execution of hand feature extraction, fusion, joints/MANO prediction, and MANO decoding, followed by scatter into fixed left/right slots.

`mano_selected_valid` and `geometry_crop_valid` gate every hand geometry loss. Mask supervision is independent and still covers both sides, including confirmed empty masks. Camera supervision retains every annotated frame. Standard evaluation/inference still predicts both hands.

Temporal acceleration requires three consecutive selected valid samples of the same side; it neither joins different hands nor bridges missing selections. Shape consistency uses only the selected valid samples of each side.

## Diagnostics

Log `accepted_samples`, `rejected_samples`, `optimizer_updated`, `sample_objectives`, `sample_replay_count`, and `gradient_rejected_sample_indices` separately. `objective` is the accepted-sample mean. After isolated gradient fallback, component diagnostics may describe the last accepted sample; `diagnostics_scope` identifies this explicitly. `forward_ms` includes loss and replay; memory snapshots are taken after backward, as identified by the log flags.
