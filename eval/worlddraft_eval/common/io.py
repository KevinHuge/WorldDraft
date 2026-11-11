"""Shared helpers for the evaluation stages: file hashes, CSV/JSON output, video sampling and benchmark paths."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

from worlddraft_eval.common import paths

ROOT = paths.REPO


def module_file(name):
    """Source file of a worlddraft_eval module, e.g. module_file('common.judge'), without importing it."""
    import importlib.util
    return Path(importlib.util.find_spec('worlddraft_eval.' + name).origin)
CELLS = [f"P{p}_{a}" for p in range(3) for a in ("AN", "AA", "AC")]


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in row.items()})


def sample_video(path, times):
    """Nearest presentation timestamps; tolerate the usual last-frame endpoint only."""
    import av
    frames, pts = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError(f"Missing frame timestamp: {path}")
            pts.append(float(frame.time))
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"Empty video: {path}")
    pts = [t - pts[0] for t in pts]
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError(f"Nonmonotonic video timestamps: {path}")
    selected = []
    for t in times:
        index = min(range(len(pts)), key=lambda i: abs(pts[i] - t))
        if abs(pts[index] - t) > 1 / fps + 1e-5:
            raise ValueError(f"Video cannot cover GT time {t}s: {path}")
        selected.append((frames[index], index, pts[index]))
    return selected, fps


def condition_video(case, cell):
    return paths.draft_video(case, cell)


def output_video(model, case, cell):
    from worlddraft_eval.common.model_config import get
    return condition_video(case, cell) if get(model).get("video") == "condition" else videos_root() / model / case / cell / "output.mp4"


def videos_root():
    """Generated videos laid out as <model>/<case>/<cell>/output.mp4; WORLDDRAFT_VIDEOS overrides the repo's results/."""
    return Path(os.environ.get("WORLDDRAFT_VIDEOS", ROOT / "results"))


def repo_path(path):
    """Local path for a path recorded in a data or protocol file (see paths.resolve)."""
    return paths.resolve(path)
