<div align="center">

<img src="assets/worlddraft-logo.png" width="480" alt="WorldDraft">

## If Code Builds the World, Can Video Models Just Render It?

Zhiyu Hou<sup>1,\*</sup>, Wen-Jie Shu<sup>2,\*</sup>, Harold H. Chen<sup>3,\*,†</sup>, Weilin Ruan<sup>4</sup>, Chenfei Liao<sup>5</sup>,<br>
Liuzhou Zhang<sup>3</sup>, Disen Lan<sup>6</sup>, Xiaogang Xu<sup>7</sup>, Qi Chen<sup>2</sup>, Ser-Nam Lim<sup>8</sup>

<sup>1</sup> SUSTech &nbsp; <sup>2</sup> ZODA &nbsp; <sup>3</sup> HKUST &nbsp; <sup>4</sup> CUHK &nbsp; <sup>5</sup> HKUST(GZ) &nbsp; <sup>6</sup> FDU &nbsp; <sup>7</sup> ZJU &nbsp; <sup>8</sup> UCF

<sup>\*</sup> Equal Contribution &nbsp; <sup>†</sup> Project Lead

<!-- Replace the Paper href (#) with the arXiv URL when available. -->
[Paper](#) &nbsp; | &nbsp; [Project Page](https://haroldchen19.github.io/WorldDraft-site/) &nbsp; | &nbsp; [Dataset](https://huggingface.co/datasets/KevinHuge/WorldDraft) &nbsp; | &nbsp; [BibTeX](#citation)

</div>

<p align="center"><img src="assets/performance.webp" width="100%" alt="WorldDraft overview: six dynamic challenge categories and a 3 × 3 grid of draft granularity and appearance alignment, yielding 432 paired cases across 48 worlds."></p>

## Overview

**WorldDraft benchmarks video models as generative renderers:** executable 4D world → draft video → photorealistic video. It tests whether models preserve the draft's geometry, motion, events, and camera while realizing the target appearance specified in text.

<p align="center"><img src="assets/overview.webp" width="100%" alt="WorldDraft overview: six dynamic challenge categories and a 3 × 3 grid of draft granularity and appearance alignment, yielding 432 paired cases across 48 worlds."></p>

- **48 worlds, six challenges:** multi-entity orchestration, multi-stage dynamics, occlusion and reappearance, camera motion, non-rigid deformation, and hybrid interactions.
- **432 paired cases:** a 3 × 3 design crosses draft granularity (abstract, simulation, previs) with appearance alignment (neutral, aligned, conflicting), holding dynamics and the target prompt fixed.
- **12 video models:** world preservation and visual realization are evaluated through Rendering Performance, Granularity Gains, and Conflict Robustness.

See the [project page](https://haroldchen19.github.io/WorldDraft-site/) for the leaderboard and video comparisons.

## Getting started

Follow the [evaluation guide](eval/USAGE.md) to set up environments and model weights, then download the benchmark:

```bash
bash eval/download_data.sh
```

Generate videos from `data/drafts` and the accompanying target prompts, saving each result as `<videos>/<model>/<case>/<cell>/output.mp4`. Register your model in [`eval/configs/models.yaml`](eval/configs/models.yaml), then run:

```bash
bash eval/eval.sh --videos <videos> --models <model> --out eval/results/<model>
```

| Resource | Contents |
|---|---|
| [Evaluation](eval/README.md) | Metrics, dependencies, and package layout |
| [Protocol](eval/docs/metrics.md) | Metric definitions, sampling, and aggregation |
| [Data guide](eval/docs/data.md) | Drafts, ground truth, and Blender scenes |

## Citation

If you find WorldDraft useful, please cite:

```bibtex
@misc{hou2026worlddraft,
  title={WorldDraft: If Code Builds the World, Can Video Models Just Render It?},
  author={Hou, Zhiyu and Shu, Wen-Jie and Chen, Harold H. and Ruan, Weilin and Liao, Chenfei and Zhang, Liuzhou and Lan, Disen and Xu, Xiaogang and Chen, Qi and Lim, Ser-Nam},
  year={2026},
  howpublished={arXiv preprint},
  url={https://haroldchen19.github.io/WorldDraft-site/}
}
```
