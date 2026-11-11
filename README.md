<h2 align="center">WorldDraft: If Code Builds the World, Can Video Models Just Render It?</h2>

<div align="center">

[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/KevinHuge/WorldDraft)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

### Table of Contents
- [News](#news)
- [TODO](#todo)
- [Overview](#overview)
- [Benchmark](#benchmark)
- [Installation](#installation)
- [Evaluation](#evaluation)
- [Repository Layout](#repository-layout)
- [Citation](#citation)

<a name="news"></a>
## 📌 News
- [11/2026] 🔥 We release the WorldDraft benchmark data and evaluation code.

<a name="todo"></a>
## 🧰 TODO
- [x] Release benchmark data.
- [x] Release evaluation code.
- [ ] Release paper.

<a name="overview"></a>
## 🌟 Overview

Code-for-world pipelines let a coding agent build the scene, motion, interactions and camera of a video in an executable
simulator such as Blender; a video model then turns the low-fidelity draft into a photorealistic video. In this setting
the video model acts as a **generative renderer**: executable world → draft video → photorealistic video.

**WorldDraft** evaluates whether video models render such drafts faithfully: whether they keep the geometry, motion,
events and camera of the draft while producing the target appearance requested in text.

- **48 scene families** in six groups: multi-entity choreography (A), multi-stage dynamics (B), occlusion and
  reappearance (C), camera–world coupling (D), articulated / non-rigid dynamics (E) and mixed scenes (F), each built as
  one Blender world.
- **3 × 3 drafts per family**: polish `P0` (proxy) / `P1` (simulation) / `P2` (previs) × appearance alignment
  `AN` (neutral) / `AA` (aligned) / `AC` (conflicting with the target text); 432 draft videos in total.
- **Ground truth exported from the Blender scenes**: visible-instance masks, projected entity boxes, event boundaries
  and camera trajectories.

<a name="benchmark"></a>
## 📊 Benchmark

| dimension | metrics |
|---|---|
| Entity & spatial | Global / Local Count, BBox IoU |
| Motion & event | Matched Track Coverage, Interaction Success Rate (ISR), Timing Coverage |
| Camera | ViPE rotation / translation error, CamMC (D series) |
| Render fidelity | attribute QA on color, material, texture, environment and lighting (RF); image quality, aesthetic quality, CLIP alignment |
| Appearance persistence | target attributes kept over time (AP) |
| Disentanglement | source color leakage in conflict cells (SL) |

Definitions, sampling and aggregation: [eval/docs/metrics.md](eval/docs/metrics.md). Data layout and ground-truth
formats: [eval/docs/data.md](eval/docs/data.md).

<a name="installation"></a>
## 🚀 Installation

```bash
git clone https://github.com/KevinHuge/WorldDraft.git
cd WorldDraft
bash eval/download_data.sh                # data/drafts and data/gt; add --with-scenes for the Blender scenes
```

Evaluation uses separate environments for the judge (Qwen3.6-27B on vLLM), tracking (SAM 3 / SAM 2), image quality and
ViPE; see [eval/USAGE.md](eval/USAGE.md#1-environments).

<a name="evaluation"></a>
## 📍 Evaluation

1. Condition your model on each draft `data/drafts/<case>/<case>_<cell>/<case>_<cell>.mp4` with the scene's target
   prompt `data/drafts/<case>/target_prompt.txt`, and store the outputs as

   ```
   <videos>/<model>/<case>/<cell>/output.mp4
   ```

2. Register how the model receives the 832×480 draft in [eval/configs/models.yaml](eval/configs/models.yaml).

3. Run all stages and build the report:

   ```bash
   bash eval/eval.sh --videos <videos> --models <model> --out eval/results/<model>
   ```

Per-stage commands and output formats: [eval/USAGE.md](eval/USAGE.md).

<a name="repository-layout"></a>
## 📁 Repository Layout

| path | content |
|---|---|
| [eval/](eval/README.md) | evaluation package, protocol files, configs, requirements, documentation |
| [data/](data/README.md) | benchmark data, downloaded from Hugging Face (not tracked) |

<a name="citation"></a>
## 📝 Citation

Coming soon.
