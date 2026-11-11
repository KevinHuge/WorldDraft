"""Freeze the videos to evaluate: <videos>/<model>/<case>/<cell>/output.mp4 (draft comes from the data directory)."""
import json
import os
from pathlib import Path

from worlddraft_eval.common import videos as B, model_config
from worlddraft_eval.pipeline.config import ROOT


def build(videos, models, out):
    """Write <out>/inventory.json for the requested models; missing cells are kept with status=missing."""
    for model in models:
        model_config.get(model)
    out = Path(out)
    path = out / 'inventory.json'
    if path.exists():
        rows = json.loads(path.read_text())['rows']
        if sorted({r['model'] for r in rows}) != sorted(models):
            raise ValueError(f'{path} is frozen for {sorted({r["model"] for r in rows})}; use a new --out')
        return path
    rows, duplicates = B.discover(videos)
    found = {r['model'] for r in rows}
    rows = [r for r in rows if r['model'] in models]
    for model in models:
        if model not in found:
            rows += [dict(model=model, case=case, cell=cell, polish=cell[:2], alignment=cell[3:], video='', status='missing',
                          source='', size_bytes=None, mtime_ns=None) for case in B.CASES for cell in B.CELLS]
    summary = [dict(model=m, expected=sum(r['model'] == m for r in rows),
                    available=sum(r['model'] == m and r['status'] == 'available' for r in rows),
                    missing=sum(r['model'] == m and r['status'] == 'missing' for r in rows)) for m in sorted(models)]
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(schema_version=1, videos=str(Path(videos).resolve()), summary=summary,
                                    duplicates=duplicates, rows=rows), indent=2) + '\n')
    B.write_csv(out / 'inventory.csv', rows)
    return path
