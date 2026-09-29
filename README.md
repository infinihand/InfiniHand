# InfiniHand: Streaming World-Space Hand Motion Estimation from Egocentric Video

[![Project Website](https://img.shields.io/badge/InfiniHand-Website-4CAF50?logo=googlechrome&logoColor=white)](https://infinihand.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2609.35743-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.35743)
[![Demo](https://img.shields.io/badge/Video-Demo-547354)](https://infinihand.github.io/#demo)

[Kerui Ren](https://scholar.google.com/citations?user=5kW5apkAAAAJ), Kaiwen Song, Weiguang Zhao, Yuxi Wang, Yufei Liu, [Bo Dai](https://daibo.info/), Haoyu Guo, Chunhua Shen, [Mulin Yu](https://scholar.google.com/citations?user=w0Od3hQAAAAJ), [Tao Lu](https://scholar.google.com/citations?user=Ch28NiIAAAAJ), Junting Dong

## Overview

InfiniHand is a streaming feed-forward framework that jointly estimates hand locations, MANO parameters, and camera trajectories from egocentric video. Built on a pretrained streaming 3D foundation model, it combines persistent geometric context with hand-centered appearance for accurate hand motion reconstruction in world coordinates. We curate approximately 5,000 hours of public egocentric video into a training corpus and train the model in two stages, improving reconstruction and generalization to in-the-wild interactions. InfiniHand achieves 11.19 FPS prediction throughput, more than twice that of HaWoR.

![Overview of InfiniHand: hand localization, camera-space reconstruction, streaming hand–camera estimation, and sparse geometric refinement.](assets/method.png)

**Pipeline of InfiniHand.** Stage I learns hand localization and camera-space reconstruction from full-image geometric features and hand-centered appearance. Stage II jointly estimates hands and camera trajectories with streaming geometric memory, transforming the reconstructed hands into a shared world frame. Sparse bundle adjustment refines camera trajectories to support long-sequence consistency.

See the [project page](https://infinihand.github.io/) for the [interactive demo](https://infinihand.github.io/#demo), [qualitative comparisons](https://infinihand.github.io/#results), and [citation](https://infinihand.github.io/#citation).
