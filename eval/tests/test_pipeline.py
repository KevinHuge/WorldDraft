"""worlddraft_eval.pipeline: inventory, per-stage runner and the long per-video format."""
import csv
import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from worlddraft_eval.pipeline import inventory, normalize, runner


def save(p, name, data):
    f = p/name; f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(data))


def rows(models, cases, status='available'):
    return [dict(model=m, case=c, cell=cell, status=status) for m in models for c in cases
            for cell in [f'P{p}_{a}' for p in range(3) for a in ('AN', 'AA', 'AC')]]


def run(p, stage, gpus=(4, 5, 6, 7)):
    with patch.object(runner.subprocess, 'Popen', return_value=Mock(pid=1, wait=lambda: 0)) as popen, \
         patch.object(runner.subprocess, 'check_output', return_value='0'):
        state = runner.run_stage(p, stage, gpus)
    return state, popen


def test_qa_stage_places_one_judge_per_gpu_and_checks_completion(tmp_path):
    save(tmp_path, 'inventory.json', dict(rows=rows(['Wan-3.0'], ['A01', 'D01'])))
    save(tmp_path, 'qa/plan.json', {})
    save(tmp_path, 'qa/status.json', dict(complete_videos=18, pending_answers=0, parse_errors=0))
    state, popen = run(tmp_path, 'qa')
    assert state['status'] == 'complete'
    workers = [c for c in popen.call_args_list if c.kwargs['env']['CUDA_VISIBLE_DEVICES']]
    assert [c.kwargs['env']['CUDA_VISIBLE_DEVICES'] for c in workers] == ['4', '5', '6', '7']
    assert all(c.args[0][c.args[0].index('--workers')+1] == '4' for c in workers)


def test_incomplete_stage_is_failed(tmp_path):
    save(tmp_path, 'inventory.json', dict(rows=rows(['Wan-3.0'], ['A01'])))
    save(tmp_path, 'qa/plan.json', {})
    save(tmp_path, 'qa/status.json', dict(complete_videos=3, pending_answers=6, parse_errors=0))
    assert run(tmp_path, 'qa')[0]['status'] == 'failed'


def test_subset_without_d_series_skips_camera_and_scopes_quality(tmp_path):
    save(tmp_path, 'inventory.json', dict(rows=rows(['MiniMax', 'draft'], ['A07'])))
    assert run(tmp_path, 'camera')[0]['status'] == 'not_applicable'
    state, popen = run(tmp_path, 'quality', (0, 1))
    prepare = popen.call_args_list[0].args[0]
    assert prepare[prepare.index('--cases')+1:prepare.index('--workers')] == ['A07']
    assert prepare[prepare.index('--workers')+1] == '2'


def test_videos_root_is_passed_to_stage_processes(tmp_path):
    save(tmp_path, 'inventory.json', dict(videos='/data/videos', rows=rows(['MiniMax'], ['A01'])))
    save(tmp_path, 'qa/plan.json', {})
    save(tmp_path, 'qa/status.json', dict(complete_videos=9, pending_answers=0, parse_errors=0))
    _, popen = run(tmp_path, 'qa')
    assert {c.kwargs['env']['WORLDDRAFT_VIDEOS'] for c in popen.call_args_list} == {'/data/videos'}


def test_inventory_marks_missing_cells_and_rejects_unknown_models(tmp_path):
    video = tmp_path/'videos/MiniMax/A01/P1_AN/output.mp4'; video.parent.mkdir(parents=True); video.write_bytes(b'x')
    path = inventory.build(tmp_path/'videos', ['MiniMax'], tmp_path/'out')
    data = json.loads(path.read_text())['rows']
    assert len(data) == 432 and sum(r['status'] == 'available' for r in data) == 1
    with pytest.raises(KeyError, match='models.yaml'):
        inventory.build(tmp_path/'videos', ['NoSuchModel'], tmp_path/'out2')


def test_normalize_qa_row_and_camera_reference(tmp_path):
    (tmp_path/'qa_per_video.csv').write_text('model,case,cell,RF_total,RF_total_status,AP_total,AP_total_status\n'
                                             'M,A01,P1_AN,0.5,complete,,pending\n')
    (tmp_path/'camera').mkdir()
    (tmp_path/'camera/video_metrics.csv').write_text('model,case,cell,status,RotErr_deg_mean,TransErr_rel_mean,CamMC_rel_mean\n'
                                                     'draft,D01,P2_AA,complete,0,0,0\ndraft,D01,P1_AA,complete,1.5,2,3\n')
    normalize.export(tmp_path, 'qa'); normalize.export(tmp_path, 'camera')
    qa = {(r['metric']): r for r in csv.DictReader((tmp_path/'qa/per_video.csv').open())}
    assert qa['RF_total']['value'] == '0.5' and qa['AP_total'] == dict(qa['AP_total'], value='', status='pending')
    camera = list(csv.DictReader((tmp_path/'camera/per_video.csv').open()))
    assert {r['cell'] for r in camera} == {'P1_AA'} and len(camera) == 3
