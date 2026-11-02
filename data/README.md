# Training data

## Datasets

| Stage | Branch | Datasets | Clip length |
|---|---|---|---|
| Stage I | MANO, Mask | ARCTIC, EgoPro, EgoStandard, H2O, HOT3D, OakInk2 | 4 frames |
| Stage II | MANO, Mask | ARCTIC, EgoPro, EgoStandard, H2O, HOT3D, OakInk2 | 36 frames |
| Stage II | Camera | ARCTIC, EgoPro, EgoPro Body, EgoStandard, EgoStandard Body, EgoVerse, HOT3D | 36 frames |

These datasets are selected by `dataset_jsons` in [the training configurations](../configs/training/). MANO/Mask use `egodemo_egopro_standard.json` and `egodemo_egostandard.json` for EgoPro and EgoStandard; Camera uses separate standard/body manifests.

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
│       ├── egodemo_egopro_standard.json
│       ├── egodemo_egostandard.json
│       ├── h2o.json
│       ├── hot3d.json
│       ├── oakink2.json
│       ├── egopro.json
│       ├── egopro_body.json
│       ├── egostandard.json
│       ├── egostandard_body.json
│       └── egoverse.json
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
