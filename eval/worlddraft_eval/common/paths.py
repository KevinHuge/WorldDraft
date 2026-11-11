"""Locations of the benchmark data, protocol files and configs.

The data directory (GT, draft videos, Blender scenes) is downloaded separately; WORLDDRAFT_DATA overrides its default
<repo>/data. Records inside the data and protocol files store paths as 'data/...' (inside the data directory) or
repo-relative paths; `resolve` turns either form, or an absolute path recorded on another machine, into a local path.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / 'eval'
PROTOCOL = EVAL / 'protocol'
CONFIGS = EVAL / 'configs'
RESULTS = EVAL / 'results'
DATA = Path(os.environ.get('WORLDDRAFT_DATA', REPO / 'data')).resolve()
GT = DATA / 'gt'
DRAFTS = DATA / 'drafts'
SCENES = DATA / 'scenes'
CASES = [f'{s}{i:02}' for s in 'ABCDEF' for i in range(1, 9)]
CELLS = [f'P{p}_{a}' for p in range(3) for a in ('AN', 'AA', 'AC')]


def has_data():
    """True when the benchmark data (eval/download_data.sh) is present."""
    return GT.is_dir() and DRAFTS.is_dir()


def draft_dir(case, cell):
    return DRAFTS / case / f'{case}_{cell}'


def draft_video(case, cell):
    return draft_dir(case, cell) / f'{case}_{cell}.mp4'


def preprocessing(case, cell='P2_AA'):
    return draft_dir(case, cell) / 'preprocessing.json'


def scene_blend(case, cell='P2_AA'):
    return SCENES / case / f'{case}_{cell}.blend'


def target_prompt(case):
    return SCENES / case / 'target_prompt.txt'


def resolve(recorded):
    """Local path for a path recorded in a data or protocol file."""
    path = Path(recorded)
    if path.is_absolute():
        if path.exists():
            return path
        parts = path.parts
        for anchor in ('data', 'eval', 'results'):
            if anchor in parts:
                return resolve(Path(*parts[parts.index(anchor):]))
        return path
    text = path.as_posix()
    if text == 'data' or text.startswith('data/'):
        return DATA.joinpath(*Path(text).parts[1:])
    return REPO / text


def settings_file(settings_path, value):
    """A file named in a protocol settings JSON: 'data/...' lives in the data directory, anything else is relative to
    the settings file."""
    return (resolve(value) if value.startswith('data/') else Path(settings_path).parent / value).resolve()


def record(path):
    """Path to store in a data or protocol record: 'data/...' inside the data directory, else repo-relative."""
    path = Path(path).resolve()
    for root, prefix in ((DATA, 'data'), (REPO, None)):
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        return f'{prefix}/{rel}' if prefix else rel
    return str(path)
