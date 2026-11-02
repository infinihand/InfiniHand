"""Approved left-MANO shape bases, shared by all dataset splits and I/O profiles."""

POLICY_VERSION = "dataset_source_v1_20261015"
DATASET_BASES = {
    "arctic": "A", "dexycb": "A", "hoi4d": "A", "ho3d": "A",
    "hot3d": "B", "h2o3d": "B", "egoverse": "B", "egodex": "B",
    "egopro": "B", "egopro_body": "B",
    "egostandard": "B", "egostandard_body": "B",
}


def basis_for(dataset_name):
    name = str(dataset_name).lower().replace("-", "_").removesuffix("_test")
    if name not in DATASET_BASES:
        raise ValueError(f"No approved MANO basis for dataset {dataset_name!r}")
    return DATASET_BASES[name]


def resolve_basis(dataset_name, metadata, task="full"):
    """Do not relabel stored joints/masks when selecting the decoder basis.

    Camera-only training does not consume MANO, joints, masks or their boxes.
    The compact-v10 exporter used basis B before recording a per-scene policy;
    unmarked v10 data can therefore serve only an explicitly approved B dataset.
    A datasets need regenerated labels and an explicit migration receipt in their
    metadata. Setting the manifest's geometry-ready flag is insufficient.
    """
    if metadata.get("mano_basis_policy") == "egohandbench_right_mirror_v3_to_native_B":
        if metadata.get("mano_left_basis") != "B" or metadata.get("schema_version") != "compact_camera_mano_joints_v9_20261012":
            raise ValueError("Inconsistent EgoHandBench v3 basis provenance")
        return "B"
    expected = basis_for(dataset_name)
    if task == "camera":
        return expected
    stored = metadata.get("mano_left_basis")
    version = metadata.get("mano_basis_policy")
    if stored is None and expected == "B" and str(metadata.get("schema_version", "")).startswith("compact_virtual_camera_v10"):
        return expected
    if stored != expected or version != POLICY_VERSION:
        raise ValueError(
            f"{dataset_name}: expected MANO basis {expected} with policy {POLICY_VERSION}; "
            f"metadata declares basis={stored!r}, policy={version!r}. "
            "Re-export camera MANO/joints, projected masks, bbox/validity and metadata "
            "together before hand-supervised training; changing a flag alone is unsafe."
        )
    return expected
