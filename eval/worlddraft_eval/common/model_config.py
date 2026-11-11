"""Per-model input handling from eval/configs/models.yaml (the only place model differences live)."""
from functools import lru_cache
from pathlib import Path

import yaml

PATH = Path(__file__).resolve().parents[2] / 'configs/models.yaml'


@lru_cache(maxsize=None)
def load():
    with PATH.open() as f:
        return yaml.safe_load(f)


def models():
    return list(load())


def get(model):
    try:
        return load()[model]
    except KeyError:
        raise KeyError(f'Unknown model {model!r}: add it to eval/configs/models.yaml') from None


def reference():
    return next(m for m, c in load().items() if c.get('role') == 'reference')


def reference_cell():
    return get(reference())['camera_reference_cell']


def is_camera_reference(model, cell):
    return model == reference() and cell == reference_cell()
