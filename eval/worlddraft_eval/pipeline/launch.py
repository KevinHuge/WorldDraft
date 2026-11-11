"""Run one evaluation module's CLI: python -m worlddraft_eval.pipeline.launch [--sam2-cpu-state] <module> <args...>.

The geometry worker additionally loads the pinned SAM3 cv-utils kernel (local snapshot, no publisher check) and can keep
SAM2 tracking state in RAM to avoid GPU OOM.
"""
from __future__ import annotations

import argparse
import functools
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'eval'))
CV_UTILS = ('kernels-community/cv-utils', '50fb2dfddda16b050e60328925b4f261d946a617')  # pinned kernel revision
MODULES = ['qa.run', 'events.run', 'geometry.run', 'geometry.tracks', 'geometry.export']


def load_sam3_kernel():
    from transformers.models.sam3_video import modeling_sam3_video
    from kernels import get_local_kernel
    from worlddraft_eval.common.models import local
    modeling_sam3_video.cv_utils_kernel = get_local_kernel(local(*CV_UTILS))
    modeling_sam3_video._load_cv_utils_kernel_once()
    if modeling_sam3_video.cv_utils_kernel in (None, False):
        raise RuntimeError('Required SAM3 cv-utils kernel did not load')


def keep_sam2_state_on_cpu():
    from transformers import Sam2VideoProcessor
    original = Sam2VideoProcessor.init_video_session

    @functools.wraps(original)
    def init(self, *args, **kwargs):
        kwargs['inference_state_device'] = 'cpu'
        return original(self, *args, **kwargs)
    Sam2VideoProcessor.init_video_session = init


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sam2-cpu-state', action='store_true', help='Store SAM2 tracking state in RAM')
    parser.add_argument('module', choices=MODULES)
    args, remaining = parser.parse_known_args()
    worker = args.module == 'geometry.run' and remaining[:1] == ['worker']
    if args.sam2_cpu_state and not worker:
        parser.error('--sam2-cpu-state requires the geometry.run worker')
    module = importlib.import_module('worlddraft_eval.' + args.module)
    sys.argv = [module.__file__, *remaining]
    if worker:
        load_sam3_kernel()
        if args.sam2_cpu_state:
            keep_sam2_state_on_cpu()
    module.main()


if __name__ == '__main__':
    main()
