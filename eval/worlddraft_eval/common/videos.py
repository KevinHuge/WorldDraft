"""Discover the videos to evaluate: generated <videos>/<model>/<case>/<cell>/output.mp4 and the draft conditions."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from worlddraft_eval.common import paths
from worlddraft_eval.common.io import videos_root

CASES = tuple(f'{series}{number:02}' for series in 'ABCDEF' for number in range(1, 9))
CELLS = tuple(f'{polish}_{alignment}' for polish in ('P0', 'P1', 'P2') for alignment in ('AN', 'AA', 'AC'))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def discover(videos: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Generated videos <videos>/<model>/<case>/<cell>/output.mp4 plus the draft conditions from the data directory."""
    candidates = {}
    models = {'draft'}
    duplicates = []

    def add(model, case, cell, video, source):
        if case not in CASES or cell not in CELLS:
            return
        models.add(model)
        if not video.is_file() or not video.stat().st_size:
            return
        key = (model, case, cell)
        record = dict(video=str(video.resolve()), source=source)
        if key in candidates:
            if record['video'] != candidates[key]['video']:
                duplicates.append(dict(model=model, case=case, cell=cell,
                                       selected=candidates[key]['video'], excluded=record['video']))
        else:
            candidates[key] = record

    videos = Path(videos) if videos is not None else videos_root()
    for video in sorted(videos.glob('*/*/*/output.mp4')):
        add(video.parents[2].name, video.parents[1].name, video.parent.name, video, 'canonical')
    for case in CASES:
        for cell in CELLS:
            add('draft', case, cell, paths.draft_video(case, cell), 'draft')
    rows = []
    for model in sorted(models):
        for case in CASES:
            for cell in CELLS:
                item = candidates.get((model, case, cell))
                stat = Path(item['video']).stat() if item else None
                rows.append(dict(model=model, case=case, cell=cell, polish=cell[:2], alignment=cell[3:],
                                 video=item['video'] if item else '', status='available' if item else 'missing',
                                 source=item['source'] if item else '', size_bytes=stat.st_size if stat else None,
                                 mtime_ns=stat.st_mtime_ns if stat else None))
    return rows, duplicates


def load_inventory(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text())['rows']
