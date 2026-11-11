"""Export per-case scores from the published per-video results."""
import html
import math
import json
from pathlib import Path
from statistics import mean

from worlddraft_eval.report import tables as B


def main():
    source = B.read(B.OUT / 'per_video.csv')
    values = {(r['model'], r['case'], r['cell'], r['metric']): float(r['value']) for r in source}
    assert len(values) == len(source), 'Duplicate per-video score'
    assert all(math.isfinite(v) for v in values.values())
    models = B.generators(values) + [B.REFERENCE]
    cohorts = {r['metric']: set(r['cases'].split(',')) for r in B.read(B.OUT / 'all_models_coverage.csv')}
    wide, detail = [], []
    for case in B.CASES:
        for model in models:
            row = dict(case=case, model=model)
            for metric in B.METRICS:
                cells = [c for c in B.CELLS if metric != 'SL_color' or c.endswith('AC')]
                scores = [values[model, case, c, metric] for c in cells if (model, case, c, metric) in values]
                complete = len(scores) == len(cells)
                score = mean(scores) * B.scale(metric) if complete else None
                row[metric] = score
                detail.append(dict(case=case, model=model, metric=metric, value=score,
                    available_cells=len(scores), expected_cells=len(cells),
                    status='complete' if complete else 'partial' if scores else 'unavailable_or_not_applicable',
                    included_in_all_models_summary=case in cohorts[metric]))
            wide.append(row)
    assert len(wide) == 48 * len(models)
    # Reproduce the published summary on its common case cohorts.
    for summary in B.read(B.OUT / 'all_models.csv'):
        for metric in B.METRICS:
            if not summary[metric]:
                continue
            scores = [r[metric] for r in wide if r['model'] == summary['Model'] and r['case'] in cohorts[metric]]
            assert scores and all(v is not None for v in scores), (summary['Model'], metric)
            assert math.isclose(mean(scores), float(summary[metric]), abs_tol=1e-8), (summary['Model'], metric)
    B.write('per_case.csv', wide)
    B.write('per_case_coverage.csv', detail)
    fields = ['case', 'model'] + list(B.LABELS)
    labels = {'case': 'Case', 'model': 'Model', **B.LABELS}
    page = '''<!doctype html><meta charset="utf-8"><title>Scores per scene</title>
<style>body{font:14px/1.6 system-ui;margin:24px}table{border-collapse:collapse}th,td{border:1px solid #ddd;padding:7px;white-space:nowrap;text-align:right}th{background:#edf3fa;position:sticky;top:0}td:first-child,td:nth-child(2){text-align:left}select{padding:6px;margin:8px}a{margin-right:16px}</style>
<h1>Scores per scene</h1>
<p>A01–F08; the nine cells of a scene are averaged equally (SL: the three AC cells). Proportions and CLIP ×100, camera errors in their own units. ↑ higher is better, ↓ lower is better.</p>
<p>— means not applicable, missing or incomplete for that scene (never zero). Every scene a model completes is shown; the summary tables use only the scenes shared by all models, so this page cannot be averaged into them. Timing is event-boundary coverage: unobserved events count as unmatched.</p>
<p><a href="per_case.csv">All metrics (CSV)</a> · <a href="per_case_coverage.csv">Coverage (CSV)</a> · <a href="per_video.csv">Per-video scores (CSV)</a></p>
<label>Model <select id="model"><option value="">All models</option>'''
    page += ''.join(f'<option>{html.escape(m)}</option>' for m in models)
    page += '</select></label> <label>Scene <select id="case"><option value="">All scenes</option>'
    page += ''.join(f'<option>{c}</option>' for c in B.CASES)
    page += '</select></label><div style="overflow:auto;max-height:80vh"><table><thead><tr>'
    page += ''.join(f'<th>{html.escape(labels[f])}</th>' for f in fields) + '</tr></thead><tbody>'
    for row in wide:
        page += f'<tr data-model="{html.escape(row["model"], quote=True)}" data-case="{row["case"]}">'
        page += ''.join('<td>' + html.escape(row[f] if f == 'model' else str(row[f]) if f == 'case' else '—' if row[f] is None else f'{row[f]:.2f}') + '</td>' for f in fields) + '</tr>'
    page += '''</tbody></table></div><script>
const model = document.getElementById('model'), item = document.getElementById('case');
function filter() { document.querySelectorAll('tbody tr').forEach(r => {
 const matches = !model.value || r.dataset.model === model.value;
 r.hidden = !(matches && (!item.value || r.dataset.case === item.value));
}); }
model.addEventListener('change', filter); item.addEventListener('change', filter);
</script>'''
    grid_values = {}
    for (model, case, cell, metric), value in values.items():
        grid_values.setdefault(model, {}).setdefault(case, {}).setdefault(cell, {})[metric] = value * B.scale(metric)
    payload = dict(models=[m for m in models], cases=B.CASES, metrics=B.METRICS, labels=B.LABELS, values=grid_values)
    template = Path(__file__).with_name('case_grid.html').read_text()
    grid = template.replace('__GRID_DATA__', json.dumps(payload, ensure_ascii=False).replace('<', r'\u003c'))
    page = page.replace('<label>Model <select id="model">', grid + '<h2>All scenes</h2><label>Model <select id="model">', 1)
    (B.OUT / 'per_case.html').write_text(page)
    print(f'Wrote {len(wide)} case/model rows, {len(detail)} metric coverage rows; summary consistency checks passed.')


if __name__ == '__main__':
    main()
