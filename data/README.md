# Training data

## Datasets

| Data group | Datasets | Training use |
|---|---|---|
| With camera poses | ARCTIC, DexYCB, EgoPro, EgoPro Body, EgoStandard, EgoStandard Body, EgoVerse, Gen-HumanEgo, HOT3D | Stage I/II MANO and Mask; Stage II Camera |
| Without camera poses | H2O3D, HO3D, HOI4D | Stage I/II MANO and Mask |

Stage I uses 4-frame clips and Stage II uses 36-frame clips. MANO/Mask use all hand-annotated training sources; only Camera requires valid camera poses. Camera supervision uses frames with `camera_valid: true` and a valid `camera_to_world`, rather than treating missing poses as identity targets. Dataset selection is configured in [the training configurations](../configs/training/).

Evaluation uses the separate ARCTIC, HOT3D, EgoDex and HOI4D test manifests. EgoDex is evaluation-only in this data collection.

## Preprocessing

1. Extract and temporally align RGB frames, hand annotations and available camera poses. Rectify distorted images and map them to a centered pinhole camera at 518 × 378 pixels; transform intrinsics, hand geometry and camera poses consistently.
2. Convert hand annotations to camera-space MANO parameters and decode 21 joints per hand, using meters for 3D coordinates. Project the MANO meshes to produce separate left/right masks and derive bounding boxes from the masks.
3. Save frame paths, image dimensions, intrinsics, hand validity and available camera poses in `metadata.json`. Build one training manifest per dataset listing scene locations and frame counts. The dataloader samples 4-frame or 36-frame clips for the corresponding stage.

## Prepared directory structure

```text
/path/to/data/
├── manifests/
│   └── train/
│       ├── arctic.json
│       ├── dexycb.json
│       ├── egopro.json
│       ├── egopro_body.json
│       ├── egostandard.json
│       ├── egostandard_body.json
│       ├── egoverse.json
│       ├── genhumanego.json
│       ├── hot3d.json
│       ├── h2o3d.json
│       ├── ho3d.json
│       └── hoi4d.json
└── scenes/
    └── <dataset>/<scene>/
        ├── metadata.json
        ├── images/
        │   └── 000000.webp
        ├── masks/
        │   ├── left/000000.webp
        │   └── right/000000.webp
        ├── camera_mano/
        │   └── 000000.npz
        └── camera_joints/
            └── 000000.npz
```

Each manifest contains `split: train` and scene entries with `name`, `prefix` and `frame_count`. Frame filenames are resolved through `metadata.json`.

| File | Contents |
|---|---|
| `metadata.json` | Per-frame paths, `image_size_hw`, `intrinsics`, hand validity, and `camera_valid` with available `camera_to_world` poses |
| `images/*.webp` | Prepared RGB frames |
| `masks/{left,right}/*.webp` | Binary projected hand masks |
| `camera_mano/*.npz` | `params_61`: `[2, 61]` MANO parameters |
| `camera_joints/*.npz` | `joints`: `[2, 21, 3]` camera-space joints in meters |

Stored hand slots are `[left, right]`; the dataloader converts them to `[right, left]`. World-space supervision is computed from camera-space geometry and valid camera poses.
