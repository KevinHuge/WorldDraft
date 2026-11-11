# Data

The benchmark data is hosted on Hugging Face ([KevinHuge/WorldDraft](https://huggingface.co/datasets/KevinHuge/WorldDraft))
and downloaded into this folder:

```bash
bash eval/download_data.sh                # drafts/ and gt/
bash eval/download_data.sh --with-scenes  # also scenes/ (Blender scenes, ~0.6 GB)
```

Set `WORLDDRAFT_DATA` to keep the data elsewhere.

```
data/
├── drafts/   432 draft videos (832×480, 81 frames, 16 fps), preprocessing metadata, target prompts
├── gt/       ground truth exported from the scenes: visible-instance masks, projected entity boxes,
│             event boundaries, camera trajectories
└── scenes/   the 432 Blender scenes with a render script per family (optional)
```

File formats and how to regenerate the ground truth: [eval/docs/data.md](../eval/docs/data.md).
