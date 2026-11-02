"""Portable model weights with licensed MANO assets supplied locally."""

MANO_ASSET_PREFIXES = ("mano_left.", "mano_right.")


def without_mano_assets(state):
    return {k: v for k, v in state.items() if not k.startswith(MANO_ASSET_PREFIXES)}


def load_inference_weights(model, state):
    """Require every learned model tensor; populate omitted MANO assets locally."""
    combined = dict(state)
    for key, value in model.state_dict().items():
        if key.startswith(MANO_ASSET_PREFIXES) and key not in combined:
            combined[key] = value
    return model.load_state_dict(combined, strict=True)
