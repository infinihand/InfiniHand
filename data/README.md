# Prepared data manifests

Place one JSON manifest per configured dataset in `manifests/train/`. Manifests are supplied with the prepared data; this repository does not bundle scene indexes or storage credentials. Update `dataset_jsons` and the corresponding `sampling_weights` together when selecting datasets.

Each manifest declares `dataset_name`, `sampling_name`, `split` (exactly `train` for training), `data_format` (`unified_preprocessed`), a supported `schema_version`, `raw_s3_root`, `hand_num_points` (21), `hand_keypoint_layout` (`mano21`), and `scenes`. Each scene has `name`, `prefix`, and a positive `frame_count`. Prefixes refer to self-contained prepared scene directories with metadata and the matching RGB and annotations. Local paths and configured object-store paths are supported.

Use the exact schema and MANO-basis labels emitted by your data exporter. Do not relabel geometry conventions without transforming the associated MANO, joints, masks, boxes and validity together. Camera and world supervision availability must reflect the actual exported data. The loader validates geometry policy independently of split selection.

The six hand datasets in the Stage I/II templates are ARCTIC, EgoPro, EgoStandard, H2O, HOT3D and OakInk2; their configured manifest names are authoritative. Camera/joint templates select sources with trajectory supervision. A configuration is a training recipe, not a certificate that external data has passed acceptance.

Keep evaluation manifests separate. The training CLI, sampler and dataset adapter enforce the training split. Verify scene identities against the official dataset partitions before generating manifests; the loader cannot infer a partition from mislabeled RGB content.
