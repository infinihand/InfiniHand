# Inference runtime options

The default command preserves FP32 hand branches, BF16 LingBot-Map features,
4 anchor frames plus 16 dense history frames, and the existing fixed-beta,
mask-guided depth search and SG5 refinement. No model weights, MANO conventions,
mask thresholds or optimization objective are changed by the runtime switches.
Refinement assumes one identity per hand slot within a clip.

## Options

| Option | Default | Effect |
|---|---|---|
| `--branch-precision fp32/bf16` | `fp32` | Optional BF16 neural branches; geometric transforms/refinement remain FP32/FP64. BF16 is not FP16 and is a separate numerical protocol. |
| `--cuda-graphs` | off | Capture stateless image encoders at the selected precision; temporal cache remains eager. Requires additional VRAM. |
| `--fused-rope` | off | Triton rotary-position kernel preserving eager dtype rounding. |
| `--compact-kv-cache` | off | Opt-in SDPA cache update avoiding append-then-evict copies, preserving dense and historical special-token order. Unsupported cache layouts use the original update. |
| `--mask-box-backend torch/triton` | `torch` | Optional exact bounding-box reduction over the same binary masks. |
| `--postprocess-storage cpu/cpu_async/cuda/auto` | `cpu` | Place auxiliary outputs on CPU, pinned asynchronous CPU buffers, or CUDA. CUDA capture has a 2 GiB mask guard. `auto` selects from free memory. |
| `--frame-cache png/memory` | `png` | Memory mode retains resized uint8 input frames, avoiding the PNG write/read cycle. Requires base masks and `--no-ba`. |

Example (replace the paths with your own assets):

```bash
python -m infinihand.infer \
  --video /path/to/input.mp4 --intrinsics /path/to/intrinsics.json \
  --checkpoint /path/to/weights/infinihand_egobench.pt \
  --config /path/to/config.json --output /path/to/new-output \
  --no-ba --branch-precision bf16 --cuda-graphs --fused-rope \
  --compact-kv-cache --mask-box-backend triton \
  --postprocess-storage cuda --frame-cache memory
```

Triton kernels require CUDA and Triton. All new acceleration flags are optional;
select the torch/default paths for unsupported platforms. Do not run concurrent
calls on the same model or change precision on a graph-prepared model; create a
fresh model instead. The CLI records precision, runtime options, storage,
forward timings and postprocess timings in `SUCCESS.json`.

## API and validation

`infinihand.infer` retains `build_model`, `run_dual_stream`,
`prepare_cuda_graphs`, `parser`, and `main`. CLI orchestration is in `cli.py`;
capture, graph, cache, kernel and profiler helpers are in `runtime/`.
`postprocess.refine_hands_tensors` accepts complete same-device CUDA predictions,
K, and binary masks; `refine_hands` retains the NumPy interface. Both use the same
packed depth search, fixed shape and timestamp-aware SG5.

Run CPU geometry/cache checks from the repository root:

```bash
python -m unittest discover -s tests -p 'test_hand_postprocess.py'
python tests/test_runtime_cache.py
```

The packed-mask tests additionally require CUDA/Triton. Check real-video
predictions, selected depth candidates, validity, and exported joints when
changing runtime or precision. Report model forward, refinement and full
pipeline wall time separately; microkernel speedups are not end-to-end FPS.
Full benchmark accuracy/speed is not implied by short-clip smoke tests.
