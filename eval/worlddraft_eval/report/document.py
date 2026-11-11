"""The reader-facing report: world-preserving rendering, draft support and source-target disentanglement."""
import csv
import html
from datetime import datetime, timezone
from statistics import mean

from worlddraft_eval.report import tables as B, results as P

OUT = B.BASE
MODELS = []  # generators in the aggregated results; set in main()
GROUPS = [
    ('Entity & Spatial Fidelity / Motion & Event Fidelity', B.ENTITY_METRICS + B.MOTION_METRICS),
    ('Camera Fidelity', B.CAMERA_METRICS),
    ('Rendering Instruction Fidelity', B.RF),
    ('Appearance Persistence', B.AP),
]
LABELS = dict(B.LABELS)
for prefix in ('RF', 'AP'):
    for attr in ('color', 'material', 'texture', 'environment', 'lighting'):
        LABELS[f'{prefix}_{attr}'] = f'{prefix} {attr}↑'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    values = P.collect()
    MODELS[:] = B.generators(values)
    md = []
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>WorldDraft results</title>',
            '<style>body{font:15px/1.7 system-ui,sans-serif;max-width:1550px;margin:30px auto;padding:0 20px;color:#20252b}h1,h2,h3{line-height:1.35}h2{margin-top:42px;border-bottom:2px solid #d9e2ee;padding-bottom:10px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px;margin:12px 0}td,th{border:1px solid #dce1e7;padding:7px 10px;text-align:right;white-space:nowrap}td:first-child,th:first-child{text-align:left}thead{background:#eef3f8}tbody tr:nth-child(even){background:#fafbfc}.scroll{overflow:auto}details{margin:15px 0;border:1px solid #dce1e7;padding:12px}summary{cursor:pointer;font-weight:600}a{color:#24579c}p{max-width:1200px}nav{padding:14px;background:#eef3f8}@media print{details{break-inside:avoid}body{max-width:none}}</style>']

    def heading(text, level=2, anchor=None):
        md.extend(['#' * level + ' ' + text, ''])
        attr = f' id="{anchor}"' if anchor else ''
        page.append(f'<h{level}{attr}>{html.escape(text)}</h{level}>')

    def paragraph(text):
        md.extend([text, ''])
        page.append('<p>' + html.escape(text) + '</p>')

    def table(headers, rows):
        assert all(len(row) == len(headers) for row in rows)
        md.extend(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'])
        md.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
        md.append('')
        page.append('<div class="scroll"><table><thead><tr>' + ''.join('<th scope="col">' + html.escape(x) + '</th>' for x in headers) + '</tr></thead><tbody>')
        for row in rows:
            page.append('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>')
        page.append('</tbody></table></div>')

    def begin(title):
        md.extend(['<details>', '<summary>' + title + '</summary>', ''])
        page.append('<details><summary>' + html.escape(title) + '</summary>')

    def end():
        md.extend(['</details>', ''])
        page.append('</details>')

    def cases(metric, cells):
        return B.cohort(values, MODELS, metric, cells)

    def score(model, metric, cells, cohort):
        if not cohort:
            return None
        return mean(mean(values[model, case, cell, metric] for cell in cells) for case in cohort) * B.scale(metric)

    def fmt(value):
        return '—' if value is None else f'{value:.2f}'

    def absolute(metrics, cells, models=MODELS):
        cohorts = {metric: cases(metric, cells) for metric in metrics}
        headers = ['Model'] + [f'{LABELS.get(k,k)} (n={len(cohorts[k])})' for k in metrics]
        rows = []
        for model in models:
            row = [model]
            for metric in metrics:
                cohort = cohorts[metric]
                available = all((model, c, cell, metric) in values for c in cohort for cell in cells)
                row.append(fmt(score(model, metric, cells, cohort)) if available else 'N/A')
            rows.append(row)
        table(headers, rows)

    heading('WorldDraft results', 1)
    paragraph('Generated ' + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC') + '. Metric definitions: eval/docs/metrics.md.')
    page.append('<nav><a href="#q1">I. World-Preserving Rendering</a> · <a href="#q2">II. Draft Support</a> · <a href="#q3">III. Source–Target Disentanglement</a> · <a href="#appendix">Reference and full grid</a></nav>')
    paragraph(f'The main tables compare {len(MODELS)} generators, 48 families × 9 cells = 432 videos each; the draft is listed in the appendix. Each video is scored, cells are averaged within a family, then families are averaged equally. n is the number of families that every compared generator completes for that metric; all models in a table use the same families.')
    paragraph('Proportions are shown in 0–100 and differences in percentage points; CLIP is cosine ×100; ViPE Rot is in degrees, Trans and CamMC are unitless. ↑ higher is better, ↓ lower is better. — means undefined or no result; N/A means not applicable or not comparable on the shared families. No composite world-fidelity score is reported.')
    paragraph('Invalid Timing answers are re-asked twice; remaining errors stay in the coverage denominator. The full-grid table of all generators is also in tables/all_models.csv.')
    heading('I. World-Preserving Rendering', anchor='q1')
    for title, cells in [('1. Canonical operating point: P1 × AN', ['P1_AN']), ('2. Full grid: mean over the 3×3 cells', B.CELLS)]:
        heading(title, 3)
        for group, metrics in GROUPS:
            paragraph(group)
            absolute(metrics, cells)
        begin('Image quality, aesthetic quality and CLIP alignment')
        absolute(B.QUALITY, cells)
        end()
    paragraph('The canonical point uses one video per family, the full grid nine. Camera covers D01–D08 only (72 videos per generator); the draft P2_AA reference compared with itself is excluded.')

    heading('II. Draft Support', anchor='q2')
    paragraph('Alignment fixed, P0 → P1 → P2. Each cell shows P0 / P1 / P2 with (Δ01 / Δ12) in brackets, on the families complete for all three polish levels (n families, 3n videos per model). For lower-is-better errors a negative difference is an improvement.')
    for a in ('AN', 'AA', 'AC'):
        begin('Alignment = ' + a + ': low-polish fidelity and polish benefit')
        cells = [f'P{p}_{a}' for p in range(3)]
        for group, metrics in GROUPS:
            paragraph(group)
            cohorts = {k: cases(k, cells) for k in metrics}
            rows = []
            for model in MODELS:
                row = [model]
                for k in metrics:
                    v = [score(model, k, [c], cohorts[k]) for c in cells]
                    row.append('—' if not cohorts[k] else ' / '.join(map(fmt, v)) + f' ({v[1]-v[0]:+.2f} / {v[2]-v[1]:+.2f})')
                rows.append(row)
            table(['Model'] + [f'{LABELS.get(k,k)} (n={len(cohorts[k])})' for k in metrics], rows)
        end()

    heading('III. Source–Target Disentanglement', anchor='q3')
    paragraph('Polish fixed, AN / AA / AC. Each cell shows AN / AA / AC with (AA − AC) in brackets: the conflict penalty for RF, the change of world preservation under conflict for the other metrics. n is the number of families complete for all three alignments.')
    for p in range(3):
        begin(f'Polish = P{p}: target compliance, conflict penalty and world stability')
        cells = [f'P{p}_{a}' for a in ('AN', 'AA', 'AC')]
        for group, metrics in [('Target Compliance under Conflict / RF', B.RF)] + GROUPS[:2] + [('Appearance Persistence under Conflict', B.AP)]:
            paragraph(group)
            cohorts = {k: cases(k, cells) for k in metrics}
            rows = []
            for model in MODELS:
                row = [model]
                for k in metrics:
                    v = [score(model, k, [c], cohorts[k]) for c in cells]
                    row.append('—' if not cohorts[k] else ' / '.join(map(fmt, v)) + f' ({v[1]-v[2]:+.2f})')
                rows.append(row)
            table(['Model'] + [f'{LABELS.get(k,k)} (n={len(cohorts[k])})' for k in metrics], rows)
        end()
    heading('Source appearance leakage: color ↓', 3)
    cells = [f'P{p}_AC' for p in range(3)]
    cohort = cases('SL_color', cells)
    table(['Model', 'P0 AC', 'P1 AC', 'P2 AC', 'Mean', 'Families'], [[m] + [fmt(score(m, 'SL_color', [c], cohort)) for c in cells] + [fmt(score(m, 'SL_color', cells, cohort)), len(cohort)] for m in MODELS])
    paragraph('Explicit source–target color conflicts only: the share of answers that still see the source color (lower is better), on families complete for the three AC cells. Material and texture leakage are not measured.')
    heading('Conflict penalty across polish levels: common conflict colors', 3)
    common_path = B.OUT / 'common_color_conflict.csv'
    with common_path.open(encoding='utf-8-sig') as f:
        common = {r['Model']: r for r in csv.DictReader(f)}
    keys = ['CP_P0', 'CP_P1', 'CP_P2', 'deltaCP_01', 'deltaCP_12']
    table(['Model', 'CP P0', 'CP P1', 'CP P2', 'ΔCP01', 'ΔCP12', 'Families'], [[m] + [fmt(float(common[m][k])) for k in keys] + [common[m]['families']] for m in MODELS])
    paragraph('Computed on the same color questions in every polish level: colors that conflict explicitly in P0, P1 and P2, observable in all six AA/AC cells and validly answered for every compared generator and the draft. It is not a difference of RF_total over different attribute sets, and differs from the all-attribute conflict penalty above.')

    heading('Appendix: draft reference, full grid and scope', anchor='appendix')
    begin('Draft reference: canonical P1 AN (not ranked)')
    for group, metrics in GROUPS:
        paragraph(group)
        absolute(metrics, ['P1_AN'], ['draft'])
    paragraph('Tracking of the draft uses a stronger SAM 2 fallback than for generators, so it is not an upper bound under the same detection protocol.')
    end()
    begin('Full 3×3 grid: every metric per cell')
    for k in B.METRICS:
        cells = [c for c in B.CELLS if k != 'SL_color' or c.endswith('AC')]
        cohort = cases(k, cells)
        paragraph(LABELS.get(k, k) + f'; shared families n={len(cohort)}')
        table(['Model'] + cells, [[m] + [fmt(score(m, k, [c], cohort)) for c in cells] for m in MODELS])
    end()
    paragraph('Scope: global Count requires all three sampled instants to be correct; BBox IoU compares amodal projected GT boxes with tracked boxes and includes a GT-initialized SAM 2 fallback; track coverage is coverage, not path accuracy; Timing Coverage has no threshold on the error to the GT time; the camera reference is the ViPE estimate of the draft; AP gates persistence by a correct anchor. Image, aesthetic and CLIP scores are reported separately from RF.')
    paragraph('Source: the per-video scores (per_video.csv) of the aggregated result directories.')
    page.append('</html>')
    (OUT / 'result.md').write_text('\n'.join(md))
    (OUT / 'result.html').write_text('\n'.join(page))

    # Check the common-camera cohort that was empty when draft self-comparison entered filtering.
    for k in B.CAMERA_METRICS:
        assert len(cases(k, B.CELLS)) == 8
        assert all(score(m, k, B.CELLS, cases(k, B.CELLS)) is not None for m in MODELS)
    for k in B.ENTITY_METRICS + ['RF_total', 'AP_total']:
        assert len(cases(k, B.CELLS)) == 48, k
    # The report must agree with canonical_P1_AN.csv.
    with (B.OUT / 'canonical_P1_AN.csv').open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            if r['Model'] not in MODELS:
                continue
            for k in B.DF + B.RF + B.AP + B.QUALITY:
                cohort = cases(k, ['P1_AN'])
                assert abs(float(r[k]) - score(r['Model'], k, ['P1_AN'], cohort)) < 1e-8, (r['Model'], k)
    assert page.count('</details>') == sum(x.startswith('<details>') for x in page)
    print(f'Report for {len(MODELS)} generators checked against the canonical table: {OUT}')


if __name__ == '__main__':
    main()
