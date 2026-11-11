"""Compare all evaluated generators on the same complete families per metric."""
import html
from datetime import datetime, timezone
from statistics import mean

from worlddraft_eval.report import tables as B, results as P

def summary(values, models, metrics):
    rows = {m: {'Model': m} for m in models + [B.REFERENCE]}
    coverage = []
    for metric in metrics:
        cells = [c for c in B.CELLS if metric != 'SL_color' or c.endswith('AC')]
        cases = B.cohort(values, models, metric, cells)
        for model, row in rows.items():
            complete = cases and all((model, c, cell, metric) in values for c in cases for cell in cells)
            row[metric] = mean(mean(values[model, c, cell, metric] for cell in cells)
                               for c in cases) * B.scale(metric) if complete else None
        coverage.append(dict(metric=metric, families=len(cases), videos_per_model=len(cases)*len(cells),
                             cases=','.join(cases), cells=','.join(cells)))
    return list(rows.values()), coverage


def main():
    B.OUT.mkdir(parents=True, exist_ok=True)
    values = P.collect()
    models = B.generators(values)
    rows, coverage = summary(values, models, B.METRICS)
    B.write('all_models.csv', rows)
    B.write('all_models_coverage.csv', coverage)
    metrics = list(B.LABELS)
    headers = ['Model'] + [B.LABELS[k] for k in metrics]
    table = [[r['Model']] + ['—' if r[k] is None else f'{r[k]:.2f}' for k in metrics] for r in rows]
    n = {r['metric']: r['families'] for r in coverage}
    table.append(['Shared families'] + [str(n[k]) for k in metrics])
    notes = [
        'Generated ' + datetime.now(timezone.utc).isoformat(),
        f'{len(models)} generators compared on the families that all of them complete for each metric; the nine cells of a '
        'family are averaged equally, then families equally. SL uses the three AC cells. The draft is the reference, on the '
        'same families. Missing results are not counted as zero.',
        'Proportions in 0–100, CLIP as cosine ×100, camera errors in their own units. ↑ higher is better, ↓ lower is better. '
        'Coverage per metric: all_models_coverage.csv.',
    ]
    md = ['# All generators', '', *[s+'\n' for s in notes],
          '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
    md += ['| '+' | '.join(row)+' |' for row in table]
    (B.OUT/'all_models.md').write_text('\n'.join(md)+'\n')
    page = '<!doctype html><meta charset="utf-8"><title>All generators</title><style>body{font:14px/1.6 system-ui;margin:30px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:8px;white-space:nowrap;text-align:right}th:first-child,td:first-child{text-align:left}thead{background:#eef3f8}</style><h1>All generators</h1>'
    page += ''.join('<p>'+html.escape(s)+'</p>' for s in notes)
    page += '<div style="overflow:auto"><table><thead><tr>'+''.join('<th>'+html.escape(s)+'</th>' for s in headers)+'</tr></thead><tbody>'
    page += ''.join('<tr>'+''.join('<td>'+html.escape(s)+'</td>' for s in row)+'</tr>' for row in table)
    (B.OUT/'all_models.html').write_text(page+'</tbody></table></div>')
    print(B.OUT/'all_models.html')


if __name__ == '__main__':
    main()
