"""Freeze <out>/inventory.json from <videos>/<model>/<case>/<cell>/output.mp4 for the given models."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worlddraft_eval.pipeline.inventory import build

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--videos', type=Path, required=True)
    p.add_argument('--models', nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(build(a.videos, a.models, a.out))
