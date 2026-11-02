# Training data

## Datasets

| Stage | Branch | Datasets |
|---|---|---|
| Stage I / Stage II | MANO, Mask | ARCTIC, DexYCB, EgoPro, EgoPro_Body, EgoStandard, EgoStandard_Body, EgoVerse, HOT3D, H2O3D, HO3D |
| Stage II | Camera | ARCTIC, EgoPro, EgoPro_Body, EgoStandard, EgoStandard_Body, EgoVerse, HOT3D |

Camera supervision uses frames with `camera_valid: true` and a valid `camera_to_world`. Dataset manifests and sampling weights are specified in [the training configurations](../configs/training/).

| Stage | Clip length | Training steps | Initial LR | Final LR / Initial LR | Final LR |
|---|---|---|---|---|---|
| Stage I | 4 frames | 300,000 | 3e-5 | 1/30 | 1e-6 |
| Stage II | 36 frames | 100,000 | 1e-5 | 5e-2 | 5e-7 |

All branches in a stage use the same schedule: 10,000 warmup steps followed by cosine decay to the final learning rate.

Evaluation uses separate ARCTIC, HOT3D, EgoDex and HOI4D test manifests.

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
│       ├── hot3d.json
│       ├── h2o3d.json
│       └── ho3d.json
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
