# Source notice

The production logic in this handoff repository was extracted from HaWoR at
commit `579253db1787b306161e81444170178cd6628e37`. The exact source mapping is
recorded in `docs/SOURCE_MAP.md`.

External code and model assets are not vendored in this repository:

- HaWoR / detector context: `https://github.com/ThunderVVV/HaWoR`
- SAM3 code: `https://github.com/facebookresearch/sam3`
- HaWoR/WiLoR detector checkpoint: the download URL used by the HaWoR README
- SAM3 checkpoint: gated Hugging Face repository `facebook/sam3`

Recipients must follow the licenses and access terms of those upstream
projects and model repositories.
