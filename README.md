# WorldDraft

Code-for-world pipelines let a coding agent build the scene, motion, interactions and camera of a video in an executable
simulator such as Blender; a video model then turns the low-fidelity draft into a photorealistic video. In this setting
the video model acts as a **generative renderer**: executable world → draft video → photorealistic video.

WorldDraft asks whether current video models render such drafts faithfully: do they keep the geometry, motion, events
and camera of the draft while producing the target appearance requested in text?

- **48 scene families** in six groups (multi-entity choreography, multi-stage dynamics, occlusion and reappearance,
  camera-world coupling, articulated / non-rigid dynamics, mixed scenes), each built as one Blender world.
- **3 × 3 drafts per family**: polish `P0` (proxy) / `P1` (simulation) / `P2` (previs) × appearance alignment
  `AN` (neutral) / `AA` (aligned) / `AC` (conflicting with the target text); 432 draft videos in total.
- **Evaluation** of world preservation (counts, boxes, tracks, events, camera), target appearance (attribute QA,
  quality) and appearance persistence, with ground truth exported from the Blender scenes.

## Repository

| path | content |
|---|---|
| [eval/](eval/README.md) | evaluation package, protocol files, requirements, [step-by-step usage](eval/USAGE.md) |
| `data/` | benchmark data, downloaded with `bash eval/download_data.sh` (not tracked) |

## Quick start

```bash
bash eval/download_data.sh
# generate <videos>/<model>/<case>/<cell>/output.mp4 from data/drafts, register the model in eval/configs/models.yaml
bash eval/eval.sh --videos <videos> --models <model> --out eval/results/<model>
```

Environments and every option are described in [eval/USAGE.md](eval/USAGE.md).
