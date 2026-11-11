"""Automatic retries for invalid Timing answers.

A Timing row whose score is parse_error/pending after the first judgement is re-asked once with a strict
chronological boundary schema and jointly judged fine boundaries; if it is still parse_error it is re-asked
once more with the clarification prepended. Answers are never clipped or reordered.
"""
import copy
import json

import numpy as np

CLARIFICATION = (
    'Judge onset and offset together for the SAME occurrence of the SAME target. '
    'For a movement stroke, offset is the transition from that ongoing stroke to its end, '
    'not merely a frame where the target is stationary. Initial stillness BEFORE onset is NOT offset. '
    'Do not substitute another repetition, another object, or a later return movement. '
    'Follow the event-specific start and end criteria. If the event or either required boundary '
    'is absent or not visibly supported, return occurred="no" with null indices; never guess '
    'a boundary to make the interval valid.\n'
)
INVALID = ('parse_error', 'pending')


def strict_schema(row):
    fields=[b+'_index' for b in row['required_boundaries']]
    frames=sorted(row['frames'],key=lambda f:f['time_s']);ids=[f['index'] for f in frames]
    def obj(props):return dict(type='object',properties=props,required=list(props),additionalProperties=False)
    absent=obj({'occurred':{'const':'no'},**{k:{'type':'null'} for k in fields}})
    branches=[absent]
    if set(fields)=={'onset_index','offset_index'}:
        for i,idx in enumerate(ids):branches.append(obj(dict(occurred={'const':'yes'},onset_index={'const':idx},offset_index={'type':'integer','enum':ids[i:]})))
    else:branches.append(obj({'occurred':{'const':'yes'},**{k:{'type':'integer','enum':ids} for k in fields}}))
    return {'anyOf':branches}


def retry_row(E, engine, row, video, gt, directory, model, clarified):
    """Re-ask one Timing row; returns (new_row, trace). Raises if the joint answer is still invalid."""
    row = copy.deepcopy(row)
    prefix = CLARIFICATION if clarified else ''
    if clarified:
        row['prompt'] = CLARIFICATION + E.timing_prompt(row)
    for frame in row['frames']:
        if E.sha256(frame['path']) != frame['sha256']:
            raise ValueError('Evidence changed')
    row['raw_answer'] = E.infer(engine, [row], [strict_schema(row)])[0]
    row['fine'] = {}
    fields = [b+'_index' for b in row['required_boundaries']]
    parsed = E.parse_timing(row['raw_answer'], row['frames'], fields)
    trace = dict(kind='clarified' if clarified else 'strict', coarse_prompt=row['prompt'], coarse_answer=row['raw_answer'])
    if parsed['status'] == 'complete':
        merged = {}
        for boundary in row['required_boundaries']:
            center = next(f['time_s'] for f in row['frames'] if f['index'] == parsed['indices'][boundary+'_index'])
            frames = E.image_evidence(video, np.arange(max(0, center-.625), min(5, center+.625)+1e-8, .125),
                                      directory/'evidence', row['id']+'_'+boundary, gt, row['participants'], model, target=row['target'])
            merged.update({f['index']: f for f in frames})
        joint = dict(row, frames=sorted(merged.values(), key=lambda f: f['time_s']))
        joint['prompt'] = prefix + E.timing_prompt(joint)
        answer = E.infer(engine, [joint], [strict_schema(joint)])[0]
        parsed = E.parse_timing(answer, joint['frames'], fields)
        trace['joint_fine'] = dict(frames=joint['frames'], prompt=joint['prompt'], raw_answer=answer)
        if parsed['status'] not in ('complete', 'missing_event'):
            raise ValueError(parsed)
        for boundary in row['required_boundaries']:
            derived = dict(occurred='yes' if parsed['status'] == 'complete' else 'no', index=parsed.get('indices', {}).get(boundary+'_index'))
            row['fine'][boundary] = dict(frames=joint['frames'], raw_answer=json.dumps(derived),
                                         derived_from='joint_fine response; same event and simultaneous boundary judgment')
    row['timing'] = E.timing_score(row)
    trace['result'] = row['timing']
    return row, trace


def retry_invalid(E, engine, rows, video, gt, directory, model):
    """Strict retry for invalid Timing rows, then a clarified retry for those still parse_error; traces in <id>.repair.json."""
    for i, row in enumerate(rows):
        if row['kind'] != 'Timing' or E.timing_score(row)['status'] not in INVALID:
            continue
        record = dict(original=copy.deepcopy(row), attempts=[])
        for clarified in (False, True):
            if clarified and E.timing_score(rows[i])['status'] != 'parse_error':
                break
            try:
                rows[i], trace = retry_row(E, engine, rows[i], video, gt, directory, model, clarified)
                record['attempts'].append(trace)
            except Exception as exc:
                record['attempts'].append(dict(kind='clarified' if clarified else 'strict', error=repr(exc)))
        E.write_json(directory/(row['id']+'.repair.json'), record)
    return rows
