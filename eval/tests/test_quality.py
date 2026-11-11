from pathlib import Path

import numpy as np
import pytest

from worlddraft_eval.quality import run as Q


def test_grid_rejects_short_clips_and_excludes_endpoint():
    assert Q.frame_grid(16, 81) == (80, list(range(0, 80, 2)))
    keep, indices = Q.frame_grid(24, 121)
    assert keep == 120 and len(indices) == 40 and indices[-1] == 117
    for fps, n in [(16, 79), (0, 80), (float('nan'), 100), (7, 40)]:
        with pytest.raises(ValueError):
            Q.frame_grid(fps, n)


def test_text_never_truncates_and_preserves_sentences():
    import clip
    assert Q.split_clip_text('A green car. A blue boat.', clip.tokenize) == ['A green car.', 'A blue boat.']
    long = ' '.join(['green'] * 50) + ', ' + ' '.join(['blue'] * 50)
    parts = Q.split_clip_text(long, clip.tokenize)
    assert len(parts) == 2 and ' '.join(parts) == long
    with pytest.raises(ValueError):
        Q.split_clip_text(' '.join(['car'] * 150), clip.tokenize)
    with pytest.raises(ValueError):
        Q.split_clip_text('   ', clip.tokenize)


def test_scaling_and_equal_frame_clause_weights():
    sims = np.tile([.2, .6], (40, 1))
    result = Q.reduce_scores([10, 90], [2, 8], sims, 2)
    assert result == pytest.approx(dict(image_quality=.5, aesthetic_quality=.5, clip_text_visual_alignment=40))
    with pytest.raises(ValueError):
        Q.reduce_scores([10], [2, 8], sims, 2)
    with pytest.raises(ValueError):
        Q.reduce_scores([10, float('nan')], [2, 8], sims, 2)
    with pytest.raises(ValueError):
        Q.reduce_scores([10, 90], [2, 8], sims[:39], 2)


def test_case_equal_mean_and_missing_coverage():
    def row(case, value, status='complete'):
        return dict(model='test', case=case, status=status, **{m: value for m in Q.METRICS})
    rows = [row('A01', 1), row('A01', 1), row('A02', 0), row('A03', None, 'missing'), row('A04', None, 'error')]
    result = Q.summarize(rows)[0]
    assert result['value'] == .5
    assert result['scored_videos'] == 3 and result['available_videos'] == 4
    assert result['missing_videos'] == 1 and result['status'] == 'incomplete'


def test_input_changes_invalidate_cache():
    plan = dict(config_hash='config', texts={'A01': {'clauses': ['green car']}})
    row = dict(input_sha256='original', case='A01')
    old = Q.cache_key(plan, row)
    assert old != Q.cache_key(plan, dict(row, input_sha256='changed'))
    plan['texts']['A01']['clauses'] = ['blue car']
    assert old != Q.cache_key(plan, row)


def test_real_decode_scores_every_quality_frame_and_40_clip_frames(tmp_path):
    import cv2
    import torch
    path = tmp_path / 'synthetic.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 16, (64, 48))
    assert writer.isOpened()
    for i in range(81):
        writer.write(np.full((48, 64, 3), i * 3, dtype=np.uint8))
    writer.release()

    class FakeClip:
        def encode_text(self, tokens):
            return torch.ones(len(tokens), 2)
        def encode_image(self, images):
            return torch.ones(len(images), 2)

    seen = dict(image=0, aesthetic=0)
    def musiq(images):
        seen['image'] += len(images)
        return torch.full((len(images), 1), 70.)
    def aesthetic(features):
        seen['aesthetic'] += len(features)
        return torch.full((len(features), 1), 6.)
    models = dict(musiq=musiq, aesthetic=aesthetic, clip_b=FakeClip(), clip_l=FakeClip(),
                  preprocess_b=lambda im: torch.zeros(3, 48, 64), preprocess_l=lambda x: x.float())
    result = Q.score_video(path, ['A car.'], models, 'cpu', batch_size=7)
    assert seen == dict(image=80, aesthetic=80)
    assert result['n_clip_frames'] == 40 and result['frame_count'] == 81
    assert result['image_quality'] == .7 and result['aesthetic_quality'] == .6
    assert result['clip_text_visual_alignment'] == pytest.approx(100, abs=1e-4)
