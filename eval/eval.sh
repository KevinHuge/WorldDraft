#!/usr/bin/env bash
# Evaluate generated videos with the full benchmark and write per-dimension CSVs plus the aggregated report.
#   bash eval/eval.sh --videos <dir with <model>/<case>/<cell>/output.mp4> --models M1 [M2 ...] --out <result dir>
#        [--stages qa events geometry camera quality] [--gpus 4 5 6 7] [--sam2-cpu-state]
# Rerunning the same command resumes unfinished work. New models must be listed in eval/configs/models.yaml.
# Relative paths are relative to the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/eval${PYTHONPATH:+:$PYTHONPATH}"
PY=$(python3 eval/worlddraft_eval/pipeline/config.py report 2>/dev/null || echo python)
VIDEOS= OUT= MODELS=() STAGES=(qa events geometry camera quality) GPUS=() EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    --videos) VIDEOS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --models) shift; MODELS=(); while [ $# -gt 0 ] && [[ $1 != --* ]]; do MODELS+=("$1"); shift; done ;;
    --stages) shift; STAGES=(); while [ $# -gt 0 ] && [[ $1 != --* ]]; do STAGES+=("$1"); shift; done ;;
    --gpus) shift; GPUS=(); while [ $# -gt 0 ] && [[ $1 != --* ]]; do GPUS+=("$1"); shift; done ;;
    --sam2-cpu-state) EXTRA+=(--sam2-cpu-state); shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$VIDEOS" ] && [ -n "$OUT" ] && [ ${#MODELS[@]} -gt 0 ] || { sed -n 2,5p "$0"; exit 2; }
GPU_ARGS=(); [ ${#GPUS[@]} -gt 0 ] && GPU_ARGS=(--gpus "${GPUS[@]}")
"$PY" -m worlddraft_eval.make_inventory --videos "$VIDEOS" --models "${MODELS[@]}" --out "$OUT"
failed=0
for stage in "${STAGES[@]}"; do
  "$PY" -m worlddraft_eval.stages "$stage" --out "$OUT" "${GPU_ARGS[@]}" "${EXTRA[@]}" || failed=1
done
"$PY" -m worlddraft_eval.aggregate --results "$OUT" --out "$OUT/report"
exit $failed
