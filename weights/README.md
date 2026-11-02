# Model assets

Provide official `lingbot-map.pt` and `wilor_final.ckpt` for foundation initialization. Place licensed MANO model files and `mano_mean_params.npz` under `/path/to/weights/mano/`. Set inference `mano_model_path` accordingly; training expects `mano_data` inside the configured WiLoR root, which can be a symlink to that asset directory. Supply SAM3, detector and DROID weights as described in `docs/INFERENCE.md`. Export complete InfiniHand weights from your training runs. Assets and checkpoints are not bundled.
