# worlddraft_eval

The evaluation package, organised by evaluation stage. Entry points: `eval/eval.sh`, or
`python -m worlddraft_eval.<module>` with `PYTHONPATH=eval`. Each stage runs in its own environment
(see [../USAGE.md](../USAGE.md)).

| path | content |
|---|---|
| `stages.py`, `make_inventory.py`, `aggregate.py` | command-line entry points: one stage, the frozen video inventory, the report |
| `qa/` | `run.py` (prepare, resume, summarize), `engine.py` (questions, evidence frames and GT zooms, judging, scoring), `evidence.py` (GT masks mapped into the generated frame) |
| `events/` | `run.py` (event questions, two-stage boundary localization), `timing_retry.py`, `interaction.py` / `alignment.py` (answer scoring) |
| `geometry/` | `run.py` (prepare, workers, report), `tracking.py` (SAM 3, SAM 2 fallback), `entity_metrics.py`, `memory_metrics.py`, `identity.py`, `common.py`, `tracks.py` / `export.py` (per-video metrics) |
| `camera/` | `run.py` (D-series jobs), `vipe.py` (ViPE estimator and trajectory errors), `report.py` |
| `quality/` | `run.py`: MUSIQ, LAION aesthetic, CLIP alignment |
| `gt/` | ground-truth export from the Blender scenes: `visible.py` (instance masks), `entities.py` (projected boxes), `regions.py` (interaction regions), `events.py` (event boundaries), `camera.py` (camera trajectories), `contracts.py`, `boxes.py`; see [../docs/data.md](../docs/data.md) |
| `report/` | `tables.py` (all CSV tables), `document.py` (`result.md|html`), `all_models.py`, `per_case.py` (+ `case_grid.html`), `results.py` (reads `<dir>/<stage>/per_video.csv`) |
| `pipeline/` | `runner.py` (one stage over an inventory), `launch.py`, `normalize.py` (long per-video format), `inventory.py`, `config.py` |
| `common/` | `paths.py` (data / protocol locations), `model_config.py` (`eval/configs/models.yaml`), `spatial_mapping.py`, `models.py` (weights by Hugging Face id), `judge.py` (Qwen3.6-27B on vLLM), `io.py`, `videos.py`, `metrics.py` |
