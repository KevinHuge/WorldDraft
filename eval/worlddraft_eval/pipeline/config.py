"""Pipeline settings: eval/configs/envs.yaml, overridden per key by the untracked envs.local.yaml."""
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
EVAL = ROOT / 'eval'
ENVS = EVAL / 'configs/envs.yaml'
LOCAL = EVAL / 'configs/envs.local.yaml'
PATH_KEYS = ('vipe_root',)


def _merge(base, local):
    out = dict(base)
    for key, value in local.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


@lru_cache(maxsize=None)
def envs():
    settings = yaml.safe_load(ENVS.read_text())
    if LOCAL.is_file():
        settings = _merge(settings, yaml.safe_load(LOCAL.read_text()) or {})
    for key in PATH_KEYS:
        if key in settings:
            settings[key] = str(ROOT / settings[key])  # absolute values pass through unchanged
    return settings


def python(stage):
    return envs()['python'][stage]


if __name__ == '__main__':
    import sys
    print(python(sys.argv[1]))
