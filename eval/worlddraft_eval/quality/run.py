"""Resumable multi-frame MUSIQ, LAION aesthetic and sentence-averaged CLIP evaluation.

VBench preprocessing and weights; first 5 seconds; CLIP on the 8 Hz grid, averaged over the target prompt sentences.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from worlddraft_eval.common.videos import discover, write_csv
from worlddraft_eval.common import model_config

from worlddraft_eval.common import models, paths

ROOT = paths.REPO
DEFAULT_OUT = paths.RESULTS / 'quality'
MODELS = model_config.models()  # default: every model registered in eval/configs/models.yaml
METRICS = ('image_quality', 'aesthetic_quality', 'clip_text_visual_alignment')
PROTOCOL = 'vbench-quality-grid-prompt-sentences'
DURATION = 5.0


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def split_clip_text(text, tokenizer):
    """Preserve original sentences; split overlong ones at clause boundaries, never truncate."""
    def fits(value):
        try:
            tokenizer(value, truncate=False)
            return True
        except RuntimeError as exc:
            if 'too long' not in str(exc).lower() and 'context length' not in str(exc).lower():
                raise
            return False
    result = []
    for sentence in re.split(r'(?<=[.!?])\s+', text.strip().replace('\n', ' ')):
        if not sentence.strip():
            continue
        parts = [sentence] if fits(sentence) else re.split(r'(?<=[,;:])\s+', sentence)
        for part in parts:
            if not fits(part):
                raise ValueError(f'CLIP clause exceeds 77 tokens; explicit split required: {part}')
            result.append(part)
    if not result:
        raise ValueError('Empty CLIP prompt')
    return result


def frame_grid(fps, frame_count):
    if not math.isfinite(fps) or fps < 8 or frame_count < int(DURATION * fps + 1e-6):
        raise ValueError('Video must cover 5 seconds with at least 8 fps')
    keep = int(DURATION * fps + 1e-6)
    indices = [round(k / 8 * fps) for k in range(40)]
    if len(set(indices)) != 40 or max(indices) >= keep:
        raise ValueError('Video cannot cover the 40 distinct CLIP evaluation frames')
    return keep, indices


def prepare(videos, out, weights, models, workers, batch_size, cases=None):
    import clip
    if (out / 'plan.json').exists():
        raise FileExistsError('Frozen plan exists; resume workers using that plan')
    manifest = json.loads((weights / 'manifest.json').read_text())
    for name, record in manifest.items():
        if sha256(weights / name) != record['sha256']:
            raise ValueError(f'Weight checksum mismatch: {name}')
    rows, _ = discover(videos)
    rows = [r for r in rows if r['model'] in models]
    if cases is not None:
        if not set(cases) <= {r['case'] for r in rows}:
            raise ValueError('Unknown requested case')
        rows = [r for r in rows if r['case'] in cases]
    if set(models) != {r['model'] for r in rows}:
        raise ValueError('Requested model missing from inventory')
    texts = {}
    for case in sorted({r['case'] for r in rows}):
        path = paths.target_prompt(case)
        prompt = path.read_text().strip()
        texts[case] = dict(prompt=prompt, source=str(path), source_sha256=sha256(path),
                           clauses=split_clip_text(prompt, clip.tokenize))
    environment = {name: importlib.metadata.version(name) for name in
                   ('torch', 'torchvision', 'numpy', 'pyiqa', 'timm', 'openai-clip', 'decord', 'pillow')}
    config = dict(protocol=PROTOCOL, runner_sha256=sha256(__file__), weights=manifest,
                  environment=environment, duration_s=DURATION, clip_hz=8, batch_size=batch_size,
                  image_preprocessing='VBench longer: uint8 bilinear resize max side 512, antialias=False, /255',
                  aesthetic_preprocessing='VBench tensor bicubic resize/crop 224, antialias=False, CLIP normalization',
                  quality_frames='all native frames in first 5 seconds', clip_text='equal mean original prompt sentences/clauses',
                  aggregation='equal mean of cells per case, then equal mean across cases')
    config_hash = digest(config)
    available = 0
    for row in rows:
        if row['status'] == 'available':
            row['input_sha256'] = sha256(row['video'])
            row['worker'] = available % workers
            available += 1
    write_json(out / 'plan.json', dict(config=config, config_hash=config_hash, workers=workers,
                                      weights=str(weights), models=models, texts=texts, rows=rows,
                                      python=sys.executable, python_version=platform.python_version()))
    counts = Counter(r['model'] for r in rows if r['status'] == 'available')
    print(f'Prepared {available} videos: {dict(counts)}', flush=True)


def load_models(weights, device):
    import clip
    import torch
    from pyiqa.archs.musiq_arch import MUSIQ
    from torchvision import transforms as T
    clip_b, preprocess_b = clip.load(str(weights / 'clip_model/ViT-B-32.pt'), device=device, jit=False)
    clip_l, _ = clip.load(str(weights / 'clip_model/ViT-L-14.pt'), device=device, jit=False)
    musiq = MUSIQ(pretrained_model_path=str(weights / 'pyiqa_model/musiq_spaq_ckpt-358bb6af.pth')).to(device).eval()
    aesthetic = torch.nn.Linear(768, 1)
    aesthetic.load_state_dict(torch.load(weights / 'aesthetic_model/sa_0_4_vit_l_14_linear.pth',
                                         map_location='cpu', weights_only=True))
    preprocess_l = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=False),
                              T.CenterCrop(224), T.Lambda(lambda x: x.float().div(255.0)),
                              T.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
    return dict(clip_b=clip_b.eval(), preprocess_b=preprocess_b, clip_l=clip_l.eval(),
                preprocess_l=preprocess_l, musiq=musiq, aesthetic=aesthetic.to(device).eval())


def reduce_scores(image, aesthetic, similarities, expected_frames):
    arrays = [np.asarray(x, dtype=float) for x in (image, aesthetic, similarities)]
    if arrays[0].shape != (expected_frames,) or arrays[1].shape != (expected_frames,):
        raise ValueError('Missing per-frame quality scores')
    if arrays[2].ndim != 2 or arrays[2].shape[0] != 40 or arrays[2].shape[1] < 1:
        raise ValueError('CLIP requires 40 frames and at least one text clause')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('Non-finite quality scores')
    return dict(zip(METRICS, (float(arrays[0].mean() / 100), float(arrays[1].mean() / 10),
                             float(arrays[2].mean() * 100))))


def score_video(path, clauses, models, device, batch_size=8):
    import clip
    import torch
    from decord import VideoReader, cpu
    from PIL import Image
    from torchvision import transforms as T
    # Decode exactly the evaluated native frames; no temporal interpolation or lossy re-encoding.
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    fps, frame_count = float(reader.get_avg_fps()), len(reader)
    keep, clip_indices = frame_grid(fps, frame_count)
    image_scores, aesthetic_scores, similarities, observed_clip = [], [], [], []
    with torch.inference_mode():
        text = models['clip_b'].encode_text(clip.tokenize(clauses, truncate=False).to(device)).float()
        text = torch.nn.functional.normalize(text, dim=-1)
        for start in range(0, keep, batch_size):
            indices = list(range(start, min(start + batch_size, keep)))
            frames = reader.get_batch(indices).asnumpy()
            if len(frames) != len(indices):
                raise ValueError('Decoder did not return every requested frame')
            images = torch.from_numpy(frames).permute(0, 3, 1, 2)
            h, w = images.shape[-2:]
            technical = images
            if max(h, w) > 512:
                scale = 512 / max(h, w)
                technical = T.Resize((int(h * scale), int(w * scale)), antialias=False)(technical)
            image_scores.extend(models['musiq'](technical.to(device).float() / 255).reshape(-1).tolist())
            features = models['clip_l'].encode_image(models['preprocess_l'](images).to(device)).float()
            features = torch.nn.functional.normalize(features, dim=-1)
            aesthetic_scores.extend(models['aesthetic'](features).reshape(-1).tolist())
            selected = [j for j, index in enumerate(indices) if index in clip_indices]
            if selected:
                inputs = torch.stack([models['preprocess_b'](Image.fromarray(frames[j])) for j in selected]).to(device)
                features = torch.nn.functional.normalize(models['clip_b'].encode_image(inputs).float(), dim=-1)
                similarities.extend((features @ text.T).tolist())
                observed_clip.extend(indices[j] for j in selected)
    if observed_clip != clip_indices:
        raise ValueError('Incomplete CLIP time grid')
    return dict(status='complete', fps=fps, frame_count=frame_count, n_quality_frames=keep,
                n_aesthetic_frames=len(aesthetic_scores), n_clip_frames=len(similarities),
                n_clip_clauses=len(clauses), clip_frame_indices=clip_indices, clip_clauses=clauses,
                per_frame=dict(image_quality_raw=image_scores, aesthetic_quality_raw=aesthetic_scores,
                               clip_cosine_by_clause=similarities),
                **reduce_scores(image_scores, aesthetic_scores, similarities, keep))


def result_path(out, row):
    return out / 'videos' / row['model'] / row['case'] / (row['cell'] + '.json')


def cache_key(plan, row):
    return digest(dict(config=plan['config_hash'], video=row['input_sha256'], text=plan['texts'][row['case']]))


def worker(out, index, device):
    import torch
    plan = json.loads((out / 'plan.json').read_text())
    if index not in range(plan['workers']):
        raise ValueError('Invalid worker')
    if plan['config']['runner_sha256'] != sha256(__file__):
        raise ValueError('Runner changed after plan was frozen')
    for name, version in plan['config']['environment'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f'Environment changed after preparation: {name}')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    with (out / f'worker-{index}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        weights = Path(plan['weights'])
        for name, record in plan['config']['weights'].items():
            if sha256(weights / name) != record['sha256']:
                raise ValueError(f'Weight changed: {name}')
        models = load_models(weights, device)
        jobs = [r for r in plan['rows'] if r.get('worker') == index]
        counts = Counter()
        for i, row in enumerate(jobs):
            identity = '/'.join(row[k] for k in ('model', 'case', 'cell'))
            started = time.time()
            target = result_path(out, row)
            write_json(out / f'worker-{index}.json', dict(status='running', current=identity, total=len(jobs), done=i, counts=dict(counts)))
            try:
                if sha256(row['video']) != row['input_sha256']:
                    raise ValueError('Video changed since plan was frozen')
                key = cache_key(plan, row)
                if target.exists():
                    old = json.loads(target.read_text())
                    if old.get('status') == 'complete' and old.get('cache_key') == key:
                        counts['cached'] += 1
                        continue
                print(f'[{index}] {i+1}/{len(jobs)} START {identity}', flush=True)
                result = score_video(Path(row['video']), plan['texts'][row['case']]['clauses'], models, device,
                                     batch_size=plan['config']['batch_size'])
                result.update(cache_key=key)
            except Exception as exc:
                traceback.print_exc()
                result = dict(status='error', error=f'{type(exc).__name__}: {exc}')
            result.update({k: row[k] for k in ('model', 'case', 'cell', 'video', 'input_sha256')})
            result.update(protocol=PROTOCOL, config_hash=plan['config_hash'], seconds=time.time()-started)
            write_json(target, result)
            counts[result['status']] += 1
            print(f'[{index}] {identity}: {result["status"]} {result["seconds"]:.1f}s ' +
                  str({k: result[k] for k in METRICS if k in result}), flush=True)
        status = 'complete_with_errors' if counts['error'] else 'complete'
        write_json(out / f'worker-{index}.json', dict(status=status, total=len(jobs), done=len(jobs), counts=dict(counts)))
    return bool(counts['error'])


def summarize(rows):
    summary = []
    for model in sorted({r['model'] for r in rows}):
        subset = [r for r in rows if r['model'] == model]
        states = Counter(r['status'] for r in subset)
        for metric in METRICS:
            cases = defaultdict(list)
            for row in subset:
                if row['status'] == 'complete':
                    cases[row['case']].append(row[metric])
            available = len(subset) - states['missing']
            summary.append(dict(model=model, metric=metric,
                value=float(np.mean([np.mean(v) for v in cases.values()])) if cases else None,
                scored_cases=len(cases), scored_videos=states['complete'], available_videos=available,
                missing_videos=states['missing'], error_videos=states['error'], pending_videos=states['pending'],
                status='complete' if states['complete'] == available and available else 'incomplete'))
    return summary


def report(out):
    plan = json.loads((out / 'plan.json').read_text())
    rows = []
    for job in plan['rows']:
        row = {k: job[k] for k in ('model', 'case', 'cell', 'video')}
        row.update(status='pending' if job['status'] == 'available' else 'missing', **dict.fromkeys(METRICS))
        path = result_path(out, job)
        if path.exists() and job['status'] == 'available':
            result = json.loads(path.read_text())
            if result.get('config_hash') != plan['config_hash']:
                raise ValueError(f'Stale quality result: {path}')
            if result['status'] == 'complete' and result.get('cache_key') != cache_key(plan, job):
                raise ValueError(f'Stale quality result input: {path}')
            row.update(status=result['status'], **{k: result.get(k) for k in METRICS})
        rows.append(row)
    write_csv(out / 'quality_per_video.csv', rows)
    summary = summarize(rows)
    write_csv(out / 'summary.csv', summary)
    write_json(out / 'status.json', dict(counts=dict(Counter(r['status'] for r in rows)), updated_at=time.time()))
    print(json.dumps(Counter(r['status'] for r in rows)), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--videos', type=Path, help='<model>/<case>/<cell>/output.mp4 tree (default: WORLDDRAFT_VIDEOS or <repo>/results)')
    ap.add_argument('--weights', type=Path, help='default: quality_weights in eval/configs/envs.yaml')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--models', nargs='+', choices=model_config.models(), default=list(MODELS))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--cases', nargs='+')
    ap.add_argument('--batch-size', type=int, default=8)
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--worker', type=int)
    action.add_argument('--report', action='store_true')
    args = ap.parse_args()
    if args.workers < 1 or args.batch_size < 1:
        ap.error('workers and batch size must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(args.videos, args.out, args.weights or models.quality_weights(), args.models, args.workers, args.batch_size, args.cases)
    elif args.report:
        report(args.out)
    else:
        raise SystemExit(int(worker(args.out, args.worker, args.device)))


if __name__ == '__main__':
    main()
