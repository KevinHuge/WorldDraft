"""Read per-video scores from result directories (<dir>/<dimension>/per_video.csv, long format)."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# Result directories to aggregate; aggregate.py sets this before running the table builders.
RESULTS: list[Path] = []


def collect(results=None) -> dict[tuple[str, str, str, str], float]:
    values, origin = {}, {}
    for directory in map(Path, results if results is not None else RESULTS):
        for path in sorted(directory.glob('*/per_video.csv')):
            with path.open(encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    if row['status'] != 'complete':
                        continue
                    key = (row['model'], row['case'], row['cell'], row['metric'])
                    if key in values:
                        raise ValueError(f'{key} appears in both {origin[key]} and {path}')
                    values[key], origin[key] = float(row['value']), path
    if not values:
        raise FileNotFoundError(f'no per-video scores under {results if results is not None else RESULTS}')
    return values


def sources(results=None) -> list[Path]:
    return [p for d in map(Path, results if results is not None else RESULTS) for p in sorted(d.glob('*/per_video.csv'))]


def case_dir(model: str, case: str) -> Path:
    """QA run directory of one model/case: <dir>/qa/videos/<model>/<case> (inputs/ and <cell>/answers.json)."""
    for directory in map(Path, RESULTS):
        path = directory / 'qa/videos' / model / case
        if path.is_dir():
            return path
    raise FileNotFoundError(f'no QA answers for {model}/{case} in {RESULTS}')
