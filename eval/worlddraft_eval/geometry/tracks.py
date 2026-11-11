"""Export dynamic-track metrics from frozen global entity assignments."""
import argparse
import csv
from pathlib import Path

from worlddraft_eval.geometry import run as R
from worlddraft_eval.geometry.export import export


def track_summary(entity, gt):
    ids = set(R.C.evaluated_entities(gt, 'dynamic'))
    valid = {k: v for k, v in entity['per_entity'].items() if v['n_visible_instants'] > 0}
    dynamic = [v for k, v in valid.items() if k in ids]
    errors = [v['mean_center_err'] for v in dynamic if v['mean_center_err'] is not None]
    miss = float(R.np.mean([v['miss_rate'] for v in dynamic])) if dynamic else None
    visible = sum(v['n_visible_instants'] for v in dynamic)
    matched = sum(r.get('pred') is not None for v in dynamic for r in v['instants'] if r.get('gt') is not None)
    return dict(track_ade=float(R.np.mean(errors)) if errors else None, track_miss_rate=miss,
                track_matched_coverage=1 - miss if miss is not None else None,
                n_dynamic_entities=len(dynamic), n_dynamic_visible_instants=visible,
                n_dynamic_matched_instants=matched,
                track_ade_alias_of='position_error' if len(dynamic) == len(valid) else '')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def distance_status(value, eligible_entities, matched_instants):
    if value is not None:
        return 'complete'
    if eligible_entities:
        assert matched_instants == 0, 'Matched observations must have a finite distance'
        return 'no_matched_predictions'
    return 'not_applicable'


def export_tracks(inventory, runs, out, verify_inputs=False):
    # The frozen preparer/exporter remains unchanged so its plan fingerprint remains reproducible.
    out = Path(out)
    base_export = out.with_name('.geometry_base_export.csv')
    export(inventory, runs, base_export, verify_inputs)
    details = out.with_name('geometry_video_details.csv')
    with details.open() as f:
        rows = list(csv.DictReader(f))
    plans = {str(Path(run).resolve()): R.C.read_json(Path(run) / 'plan.json') for run in runs}
    ground_truth = {}
    by_key = {}
    for row in rows:
        if row['status'] == 'complete':
            path = plans[row['source_run']]['ground_truth'][row['case']]['path']
            if path not in ground_truth:
                ground_truth[path] = R.C.read_json(path)
            row.update(track_summary(R.C.read_json(row['entity_artifact']), ground_truth[path]))
        by_key[row['model'], row['case'], row['cell']] = row
    write_csv(details, rows)
    with base_export.open() as f:
        long_rows = list(csv.DictReader(f))
    appended = []
    for row in long_rows:
        detail = by_key[row['model'], row['case'], row['cell']]
        if row['metric'] == 'position_error' and detail['status'] == 'complete':
            row['status'] = distance_status(row['value'] if row['value'] != '' else None,
                int(detail['n_entities']), int(detail['n_matched_entity_instants']))
        if row['metric'] != 'track_ade':
            continue
        row.update(value=detail.get('track_ade'), coverage=detail.get('track_matched_coverage'),
                   protocol='global_assignment_dynamic_tracks')
        if detail['status'] == 'complete':
            row['status'] = distance_status(row['value'], detail['n_dynamic_entities'], detail['n_dynamic_matched_instants'])
        for metric in ('track_miss_rate', 'track_matched_coverage'):
            value = detail.get(metric)
            status = ('complete' if value is not None else 'not_applicable') if detail['status'] == 'complete' else detail['status']
            appended.append(dict(row, metric=metric, value=value, unit='ratio', status=status))
    write_csv(out, long_rows + appended)
    base_export.unlink()
    status_path = out.with_name('geometry_status.json')
    status = R.C.read_json(status_path)
    status.update(track_exporter_sha256=R.hash_file(__file__), metrics_per_video=10,
                  track_scope='GT dynamic entities, subset of unchanged global assignment; equal entity weighting')
    R.write_json(status_path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--verify-inputs', action='store_true')
    export_tracks(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
