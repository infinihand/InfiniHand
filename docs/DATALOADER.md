# Unified prepared-data loader

All datasets use `infinihand/training/data/datasets/unified_hand_dataset.py`. Source conversion belongs to offline preprocessing. Training accepts prepared 518x378 compact virtual-camera scenes with matching intrinsics; it does not resize or rectify them again.

## Stored and runtime data

- RGB and binary masks are lossless WebP. `metadata.json` describes saved image dimensions, K, camera validity, source frame IDs, timestamps and per-side hand supervision.
- File slots left/right become model slots right/left once.
- Camera MANO is [2,61]: global orientation3, pose45, shape10 and wrist position3, in meters. Camera joints [2,21,3] decode from the same MANO; joint0 must equal its wrist translation.
- World joints derive from camera joints and valid source c2w: `Jw = Rcw Jc + tcw`. No duplicate world geometry files are read during training.
- `hand_valid` gates geometric supervision, not mask pixels. Runtime `mano_valid` and joint-valid tensors are derived loss masks.
- Projected masks define boxes. MANO-only reads their exported metadata boxes; Mask/joint reads the actual masks. Camera-only reads neither.

## Approved left-MANO shape basis

`infinihand/training/data/mano_basis_policy.py` is authoritative:

| Basis | Datasets |
|---|---|
| A | ARCTIC, DexYCB, HOI4D, HO3D |
| B | HOT3D, H2O3D, EgoVerse, EgoDex, EgoPro, EgoPro-body, EgoStandard, EgoStandard-body |

Test splits inherit the same basis. Unknown datasets have no default. A uses the source left `shapedirs`; B negates its x component. Model decoding uses a functional buffer override rather than mutating shared MANO assets.

For hand-supervised profiles, metadata must match the approved basis and `mano_basis_policy=dataset_source_v1_20261015`. An unmarked compact-v10 export is accepted only for a listed B dataset because that exporter used B. A datasets require consistent regeneration of camera MANO/joints, projected masks, bbox/validity and metadata. The loader rejects unmarked or conflicting A data rather than merely swapping the decoder. Camera-only skips this check because it does not use hand labels.

## Task-specific loading and transfer

| Task | Loaded objects | Executed branches |
|---|---|---|
| Mask | RGB, projected masks, metadata | LingBot features, DPT, mask |
| MANO | RGB, camera MANO/joints, metadata bbox/K | LingBot features, crop encoder, reconstruction/joints |
| Camera | RGB, metadata K/trajectory | LingBot features, camera |
| Joint | RGB, projected masks, camera MANO/joints, metadata | All training heads |

RGB stays uint8 and binary masks bool through pinned-memory/H2D transfer, then converts to FP32 for computation. There is no precision loss for 8-bit RGB or binary masks. Interpolation and predictions stay floating point. Actual model boxes define the perspective transform used by both RGB/feature crops and 2D targets; duplicate loader-generated crop images and crop masks are absent from training batches.

## Shared metadata cache

`metadata_cache_dir` enables lossless LZ4 files under `metadata_lz4/v1/`. Production
configs use `/path/to/cache/metadata`, shared by all nodes and ranks.
Only original metadata JSON is cached, never RGB, masks, MANO or joint files.
The first access downloads and validates the manifest SHA256, then atomically
publishes the compressed file. Source URI plus manifest SHA256 form the cache key,
so revised labels cannot reuse stale metadata. POSIX locks serialize concurrent
misses and repairs; regular hits only read, decompress, verify and parse the file.

There is no persistent decoded-scene cache. Metadata and derived camera arrays
live only for the current sample, including when `data_cache_mode=none`. That
legacy setting controls preload-manifest selection, independently of the LZ4
disk cache. A blank `metadata_cache_dir` disables the disk cache as well. This
requires POSIX shared-file locking and the `lz4` dependency. The OS may cache file
pages; the application does not reserve a full in-memory copy per worker.

## Sampling and validity

Weights are `coefficient * sqrt(eligible completed frames)`: ARCTIC/HOT3D=10, other configured sources=1. Each selected dataset samples uniformly from its entire eligible scene pool. Virtual length controls dataset frequency, not scene coverage. All training tasks use manifests explicitly labeled `split: train`. Dataset weights are normalized over the training pool.

Stage-I MANO requires at least one valid hand in every sampled frame. Joint Stage II requires at least ceil(0.7*36)=26 valid-hand frames. Mask/Camera have no such quota. No invalid frame is removed or repeated to fill a window. Reconstruction selects one valid side per frame; no valid side skips the hand branch, while mask/camera losses remain available.

`ids` preserves source metadata indices; `source_frame_ids` preserves source frame IDs. Explicit unit-tagged timestamps take precedence for temporal loss; absent timestamps fall back to the metadata timeline at 0.1s per stored 10FPS frame. Sparse selection retains original time gaps.

## Mask supervision

Left/right target masks are binary projected regions. A geometry-invalid side may still have a supervised projected mask. Confirmed absence supplies a zero target; missing annotation or failed fitting is unknown and does not become an automatic negative. `mask_supervision_valid` and documented projection/absence reasons determine whether BCE/Dice is applied. `hand_valid=false` does not erase mask pixels.

## Acceptance

Verify the geometry conventions and annotation validity of every input dataset before training.
