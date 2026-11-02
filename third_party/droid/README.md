# DROID sparse BA components

This directory contains the feature/update network, depth buffer, factor graph and global BA backend used by `infinihand.sparse_ba_backend`. Poses are initialized from InfiniHand predictions. Online SLAM tracking, frontend keyframe selection, trajectory filling, viewers and training data readers are not included.

- `droid_ba/`: Python BA runtime and geometry helpers.
- `src/`: CUDA/C++ correlation and bundle-adjustment operators.
- `thirdparty/lietorch/`: Lie-group runtime and extension sources.
- `thirdparty/eigen/Eigen/`: headers required by the extensions.

Build with `python -m pip install --no-build-isolation /path/to/InfiniHand/third_party/droid`. Provide `droid.pth` separately. Upstream licenses are retained in `LICENSE` and under `thirdparty/`.
