import json

from worlddraft_eval.events import run as E, timing_retry as TR


def frames(n=4):
    return [dict(index=i, time_s=i*.5, path=f'f{i}', sha256='x') for i in range(n)]


def timing_row():
    return dict(id='e1', kind='Timing', required_boundaries=['onset', 'offset'], frames=frames(), participants=['car'],
                target='car', prompt='coarse', raw_answer='{"occurred":"yes","onset_index":3,"offset_index":1}', fine={},
                gt={'onset': dict(time_s=.5, interval_s=[.4, .6]), 'offset': dict(time_s=1.5, interval_s=[1.4, 1.6])})


def patch(monkeypatch, answers):
    calls = []
    def infer(engine, batch, schemas):
        calls.append(batch[0]['prompt']); return [answers.pop(0)]
    monkeypatch.setattr(E, 'infer', infer)
    monkeypatch.setattr(E, 'sha256', lambda p: 'x')
    monkeypatch.setattr(E, 'timing_prompt', lambda row, boundary=None: 'timing')
    monkeypatch.setattr(E, 'image_evidence', lambda video, times, *a, **k: [dict(index=10+i, time_s=float(t), path='g', sha256='x') for i, t in enumerate(times)])
    return calls


def run(tmp_path, rows):
    return TR.retry_invalid(E, object(), rows, 'video.mp4', {}, tmp_path, 'M')


def test_valid_rows_are_not_retried(tmp_path, monkeypatch):
    calls = patch(monkeypatch, [])
    row = dict(timing_row(), raw_answer='{"occurred":"no","onset_index":null,"offset_index":null}')
    assert run(tmp_path, [row]) == [row] and calls == []


def test_invalid_row_gets_strict_then_clarified_retry(tmp_path, monkeypatch):
    bad = '{"occurred":"yes","onset_index":3,"offset_index":1}'
    calls = patch(monkeypatch, [bad, bad])
    row = run(tmp_path, [timing_row()])[0]
    assert len(calls) == 2 and calls[1].startswith(TR.CLARIFICATION)
    assert E.timing_score(row)['status'] == 'parse_error'
    record = json.loads((tmp_path/'e1.repair.json').read_text())
    assert [a['kind'] for a in record['attempts']] == ['strict', 'clarified'] and all(a['result']['status'] == 'parse_error' for a in record['attempts'])


def test_strict_retry_that_resolves_skips_clarified(tmp_path, monkeypatch):
    calls = patch(monkeypatch, ['{"occurred":"no","onset_index":null,"offset_index":null}'])
    row = run(tmp_path, [timing_row()])[0]
    assert len(calls) == 1 and not calls[0].startswith(TR.CLARIFICATION)
    assert E.timing_score(row)['status'] == 'missing_event'


def test_strict_schema_offset_not_before_onset():
    schema = TR.strict_schema(timing_row())
    yes = [b for b in schema['anyOf'] if b['properties']['occurred']['const'] == 'yes']
    assert yes[2]['properties']['onset_index'] == {'const': 2} and yes[2]['properties']['offset_index']['enum'] == [2, 3]
