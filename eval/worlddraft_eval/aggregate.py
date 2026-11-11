"""Aggregate per-video CSVs of one or more result directories into the benchmark report.

Writes <out>/tables/*.csv (per_video, grid_<metric>, full_grid, canonical_P1_AN, all_models, per_case, ...),
<out>/tables/tables.zip and the reading report <out>/result.md|html.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worlddraft_eval.report import (results as published, tables, all_models as summary, document as report, per_case as cases)


def main(results, out):
    published.RESULTS[:] = [Path(r).resolve() for r in results]
    out = Path(out).resolve()
    tables.BASE, tables.OUT, report.OUT = out, out / "tables", out
    for builder in (summary, tables, report, cases):
        builder.main()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.results, a.out)
