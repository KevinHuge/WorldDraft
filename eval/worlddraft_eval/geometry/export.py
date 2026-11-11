"""Export per-video geometry metrics from geometry-stage runs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import time

from worlddraft_eval.geometry import run as R
from worlddraft_eval.common.spatial_mapping import input_crop


def geometry_summary(entity):
    """One entity mean per video; missing observations never get a fake distance."""
    valid = [e for e in entity['per_entity'].values() if e['n_visible_instants'] > 0]
    visible = sum(e['n_visible_instants'] for e in valid)
    matched = sum(r.get('pred') is not None for e in valid for r in e['instants'] if r.get('gt') is not None)
    summary = entity['summary']
    return dict(bbox_iou=summary['mean_iou'], position_error=summary['mean_center_err'],
                track_ade=summary['mean_center_err'], miss_rate=summary['miss_rate'],
                matched_coverage=1 - summary['miss_rate'] if summary['miss_rate'] is not None else None,
                matched_instant_coverage=matched / visible if visible else None,
                n_visible_entity_instants=visible, n_matched_entity_instants=matched,
                n_entities=summary['n_entities'], n_tracks=summary['n_tracks'], n_assigned=summary['n_assigned'])


def export(inventory, runs, out, verify_inputs=False):
    out = Path(out)
    jobs = {}
    for run in map(Path, runs):
        plan = R.C.read_json(run / 'plan.json')
        assert R.hash_value(plan['config']) == plan['config_hash']
        assert R.code_fingerprint() == plan['config']['code']
        if verify_inputs:
            for gt in plan['ground_truth'].values():
                assert R.hash_file(gt['path']) == gt['sha256']
        for job in plan['jobs']:
            key = job['model'], job['case'], job['cell']
            assert key not in jobs, key
            jobs[key] = (run, plan, job)
    result = []
    for source in R.C.read_json(inventory)['rows']:
        row = {k: source[k] for k in ('model', 'case', 'cell', 'video')}
        row['status'] = 'missing' if source['status'] != 'available' else 'pending'
        entry = jobs.get((row['model'], row['case'], row['cell']))
        if entry:
            run, plan, job = entry
            assert Path(job['video']).resolve() == Path(row['video']).resolve()
            row.update(source_run=str(run.resolve()), video_sha256=job['video_sha256'])
            path = R.artifact(run, 'metrics', job)
            if path.exists():
                metric = R.C.read_json(path)
                row['status'] = metric['status']
                if metric['status'] == 'complete':
                    key = R.hash_value(dict(config=plan['config_hash'], video=job['video_sha256'],
                        gt=plan['ground_truth'][job['case']]['sha256'], sample=job['sample']))
                    assert metric['cache_key'] == key
                    if verify_inputs:
                        assert R.hash_file(job['video']) == job['video_sha256']
                    entity = R.C.read_json(R.artifact(run, 'entity', job))
                    row.update(geometry_summary(entity))
                    assert row['position_error'] == metric['summary']['mean_center_err']
                    row.update({f'dinov2_appearance_{k}': metric['summary'][f'identity_{k}']
                                for k in ('short', 'revisit', 'cross')})
                    row['fallback_entities'] = metric['summary']['fallback_entities']
                    row['spatial_mapping'] = job['sample'].get('extra', {}).get('spatial_mapping', '')
                    row['entity_artifact'] = str(R.artifact(run, 'entity', job).resolve())
                    row['distance_units'] = 'normalized_unit_square_diagonal'
                    row['track_ade_alias_of'] = 'position_error'
                else:
                    row['error'] = metric.get('error', '')
        result.append(row)
    fields = list(dict.fromkeys(k for r in result for k in r))
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name('geometry_video_details.csv.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)
    temporary.replace(out.with_name('geometry_video_details.csv'))
    metrics = ['bbox_iou', 'position_error', 'track_ade', 'miss_rate', 'matched_coverage',
               'dinov2_appearance_short', 'dinov2_appearance_revisit', 'dinov2_appearance_cross']
    temporary = out.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'case', 'cell', 'metric', 'value', 'status',
            'unit', 'coverage', 'source', 'protocol'])
        writer.writeheader()
        for row in result:
            for metric in metrics:
                unit = ('normalized_unit_square_diagonal' if metric in ('position_error', 'track_ade')
                        else 'cosine_similarity' if metric.startswith('dinov2') else 'ratio')
                writer.writerow(dict(model=row['model'], case=row['case'], cell=row['cell'], metric=metric,
                    value=row.get(metric), status=('not_applicable' if row['status'] == 'complete' and row.get(metric) is None
                                                   else row['status']), unit=unit, coverage=row.get('matched_coverage'),
                    source=row.get('entity_artifact', row.get('source_run', '')),
                    protocol='appearance_diagnostic' if metric.startswith('dinov2')
                             else 'global_track_assignment'))
    temporary.replace(out)
    status = dict(updated_at=time.time(), counts=dict(Counter(r['status'] for r in result)),
                  source_runs=[str(Path(r).resolve()) for r in runs], verified_video_hashes=verify_inputs,
                  inventory_sha256=R.hash_file(inventory), exporter_sha256=R.hash_file(__file__))
    R.write_json(out.with_name('geometry_status.json'), status)
    print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('export')
    p.add_argument('--inventory', required=True)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--verify-inputs', action='store_true')
    args = vars(parser.parse_args())
    command = args.pop('command')
    export(**args)


if __name__ == '__main__':
    main()
