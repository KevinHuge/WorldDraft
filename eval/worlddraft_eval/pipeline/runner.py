"""Run one evaluation stage over a frozen inventory: prepare (once) -> one worker per GPU -> report -> checks.

Each stage runs worlddraft_eval modules in its own interpreter (eval/configs/envs.yaml).
Rerunning a stage resumes from its frozen plan.
"""
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from worlddraft_eval.common import paths
from worlddraft_eval.pipeline.config import EVAL, ROOT, envs, python

PLANS = {'qa': 'qa/plan.json', 'events': 'events/plan.json', 'quality': 'quality/plan.json',
         'geometry': 'geometry/plan.json', 'camera': 'camera/plan.json'}
CAMERA_GT = paths.GT / 'camera'
FINAL = ('complete', 'not_applicable', 'no_matched_events')


def inventory(out):
    data = json.loads((Path(out) / 'inventory.json').read_text())
    return data, [r for r in data['rows'] if r['status'] == 'available']


def launch(out, cmd, label, gpu=None, videos=None):
    out = Path(out)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', VLLM_NO_USAGE_STATS='1',
               TORCHINDUCTOR_CACHE_DIR=str(out / 'cache' / f'gpu-{gpu}' / 'torchinductor'),
               VLLM_CACHE_ROOT=str(out / 'cache' / f'gpu-{gpu}' / 'vllm'))
    env['PATH'] = str(Path(cmd[0]).parent) + ':' + env['PATH']
    env['PYTHONPATH'] = str(EVAL) + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if videos:
        env['WORLDDRAFT_VIDEOS'] = videos
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    with (out / 'logs' / (label + '.log')).open('a') as f:
        return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)


def command(stage, action, out, models, cases, workers, index=None, sam2_cpu_state=False):
    out = Path(out)
    py = python(stage)
    if stage == 'camera':
        c = [py, '-u', '-m', 'worlddraft_eval.camera.run', '--out', str(out / 'camera'), '--vipe-root', envs()['vipe_root'], '--' + action]
        if action == 'prepare':
            c += ['--models', *models]
        if index is not None:
            c += [str(index)]
    elif stage == 'quality':
        c = [py, '-u', '-m', 'worlddraft_eval.quality.run', '--out', str(out / 'quality'), '--' + action]
        if action == 'prepare':
            c += ['--models', *models, '--cases', *cases, '--workers', str(workers)]
        if index is not None:
            c += [str(index)]
    else:
        module = stage + '.run'
        c = [py, '-u', '-m', 'worlddraft_eval.pipeline.launch', module, action, '--out', str(out / 'geometry' if stage == 'geometry' else out)]
        if action == 'prepare':
            c += ['--workers', str(workers)]
        if stage == 'geometry' and action == 'prepare':
            c += ['--cases', *cases, '--models', *models]
        if index is not None:
            c += ['--index' if stage == 'geometry' else '--worker', str(index)]
        if index is not None and stage != 'geometry':
            c += ['--batch-size', '4']
        if stage == 'qa' and action == 'judge':
            c += ['--workers', str(workers)]
        if stage == 'geometry' and action == 'worker' and sam2_cpu_state:
            c.insert(4, '--sam2-cpu-state')  # launcher option goes before the module name
    return c


def check(out, cmd, label, videos):
    code = launch(out, cmd, label, videos=videos).wait()
    if code:
        raise RuntimeError(f'{label} exited {code}; see {Path(out) / "logs" / (label + ".log")}')


def verify(stage, out, expected, models, camera_videos):
    out = Path(out)
    if stage == 'quality':
        assert json.loads((out / 'quality/status.json').read_text())['counts'] == {'complete': expected}
    elif stage == 'qa':
        s = json.loads((out / 'qa/status.json').read_text())
        assert s.get('complete_videos') == expected and s['pending_answers'] == s['parse_errors'] == 0, s
    elif stage == 'geometry':
        assert json.loads((out / 'geometry/status.json').read_text()).get('complete') == expected
    elif stage == 'events':
        for name in ('motion_per_video.csv', 'timing_per_video.csv'):
            with (out / name).open() as f:
                assert all(r['status'] in FINAL for r in csv.DictReader(f)), name
    elif stage == 'camera':
        with (out / 'camera/video_metrics.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert sum(r['model'] in models and r['status'] == 'complete' for r in rows) == camera_videos


def run_stage(out, stage, gpus, sam2_cpu_state=False):
    """Run one stage to completion; returns its status dict (also saved to <out>/status/<stage>.json)."""
    out = Path(out).resolve()
    data, rows = inventory(out)
    models = sorted({r['model'] for r in rows})
    cases = sorted({r['case'] for r in rows})
    expected = len(rows)
    camera_videos = sum(r['case'].startswith('D') for r in rows)
    videos = data.get('videos')
    state = dict(stage=stage, gpus=list(gpus), expected_videos=expected, status='running', started_at=time.time())
    status_path = out / 'status' / f'{stage}.json'
    status_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        state['updated_at'] = time.time()
        status_path.write_text(json.dumps(state, indent=2) + '\n')
    save()
    if not rows or (stage == 'camera' and not camera_videos):
        state['status'] = 'not_applicable'
        save()
        return state
    if stage == 'camera' and len(gpus) != 4:
        raise ValueError('camera plans are fixed at 4 workers; pass 4 GPUs')
    try:
        if not (out / PLANS[stage]).exists():
            if stage == 'camera':
                (out / 'camera/gt').mkdir(parents=True, exist_ok=True)
                for gt in CAMERA_GT.glob('*.json'):
                    if not (out / 'camera/gt' / gt.name).exists():
                        shutil.copyfile(gt, out / 'camera/gt' / gt.name)
            check(out, command(stage, 'prepare', out, models, cases, len(gpus)), stage + '-prepare', videos)
        if stage == 'quality':
            assert len(json.loads((out / PLANS[stage]).read_text())['rows']) == expected
        workers = []
        for i, gpu in enumerate(gpus):
            while int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip()) > 1024:
                time.sleep(10)
            action = 'worker' if stage in ('quality', 'geometry', 'camera') else 'judge'
            workers.append(launch(out, command(stage, action, out, models, cases, len(gpus), i, sam2_cpu_state), f'{stage}-{i}', gpu, videos))
        state['pids'] = [p.pid for p in workers]
        save()
        codes = [p.wait() for p in workers]
        check(out, command(stage, 'report', out, models, cases, len(gpus)), stage + '-report', videos)
        if any(codes):
            raise RuntimeError(f'worker exit codes {codes}')
        if stage == 'geometry':
            check(out, [python('geometry'), '-m', 'worlddraft_eval.pipeline.launch', 'geometry.tracks', '--inventory', str(out / 'inventory.json'),
                        '--runs', str(out / 'geometry'), '--out', str(out / 'geometry_per_video.csv')], 'geometry-export', videos)
        verify(stage, out, expected, models, camera_videos)
        state['status'] = 'complete'
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
    save()
    return state
