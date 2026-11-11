"""Evaluate one dimension for every video in <out>/inventory.json; writes <out>/<stage>/per_video.csv.

    python -m worlddraft_eval.stages <qa|events|geometry|camera|quality> --out <result dir> [--gpus 0 1 ...]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worlddraft_eval.pipeline import normalize, runner
from worlddraft_eval.pipeline.config import envs

STAGES = ('qa', 'events', 'geometry', 'camera', 'quality')


def main(stage, out, gpus, sam2_cpu_state=False):
    state = runner.run_stage(out, stage, gpus, sam2_cpu_state)
    if state['status'] == 'complete':
        normalize.export(out, stage)
    print(stage, state['status'], state.get('error', ''), flush=True)
    return state


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=STAGES)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', type=int, default=envs()['gpus'])
    p.add_argument('--sam2-cpu-state', action='store_true', help='geometry: keep SAM2 tracking state in RAM')
    a = p.parse_args()
    sys.exit(0 if main(a.stage, a.out, a.gpus, a.sam2_cpu_state)['status'] in ('complete', 'not_applicable') else 1)
