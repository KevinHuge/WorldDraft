# Evaluation

The Python package `worlddraft_eval` scores videos rendered from the WorldDraft drafts and builds the benchmark tables.
[USAGE.md](USAGE.md) gives the commands, step by step.

## Tests

From the repository root, with Python 3.11 and `requirements.txt`:

```bash
python -m pip install -r eval/requirements.txt
python -m pytest eval/tests -q
```

Tests that read the benchmark data use `data/` (or `WORLDDRAFT_DATA`) and are skipped without it
([USAGE section 2](USAGE.md#2-data)).

## Metrics

| group | metric | definition |
|---|---|---|
| Entity & spatial | Count (global / local) | judged object count equals the visible GT count at every sampled instant (global) / per GT region (local) |
| | BBox IoU | box IoU of the tracked entities against the projected Blender boxes, after one video-level assignment |
| Motion & event | Track coverage | share of GT-valid instants at which a matched track has a box (dynamic entities) |
| | ISR | Interaction Success Rate: events whose action and outcome the judge confirms |
| | Timing coverage | share of expected events whose required boundaries the judge localizes |
| Camera | Rot / Trans / CamMC | ViPE trajectory of the video against ViPE of the reference draft (D series only); lower is better |
| Render fidelity | RF | binary attribute questions on target color, material, texture, environment and lighting |
| | Image / Aesthetic / CLIP | MUSIQ, LAION aesthetic head on CLIP ViT-L/14, frame-text CLIP similarity (x100) |
| Appearance | AP | attribute persistence, gated by a correct anchor-frame answer |
| Disentanglement | SL (color) | source color still present in conflict cells (AC); lower is better |

Scores are averaged within a scene family over the conditions of a table, then over families; every table uses the
families that all models in the table complete for that metric. The full protocol is in
[docs/metrics.md](docs/metrics.md).

## Layout

| path | content |
|---|---|
| `eval.sh` | one command: inventory, every stage, aggregation |
| `worlddraft_eval/` | the package; module map in [worlddraft_eval/README.md](worlddraft_eval/README.md) |
| `protocol/` | question bank, per-scene QA settings, event definitions, entity rules ([protocol/README.md](protocol/README.md)) |
| `configs/` | `models.yaml` (how each model receives the 832x480 draft), `envs.yaml` (interpreters, GPUs, weights) |
| `docs/` | [metrics.md](docs/metrics.md) (definitions, sampling, aggregation), [data.md](docs/data.md) (data layout, GT formats, regenerating GT) |
| `results/` | evaluation outputs (not tracked) |
