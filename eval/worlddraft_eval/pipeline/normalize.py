"""Turn each stage's native per-video CSV into the common long format: model,case,cell,metric,value,status."""
import csv
import math
from pathlib import Path

from worlddraft_eval.common import model_config
from worlddraft_eval.common.metrics import CAMERA, QA_KEYS

QUALITY = ['image_quality', 'aesthetic_quality', 'clip_text_visual_alignment']
GEOMETRY = ['bbox_iou', 'track_matched_coverage']
FIELDS = ['model', 'case', 'cell', 'metric', 'value', 'status']
# Native outputs of each stage, relative to the run directory, with the parser group for each file.
SOURCES = {
    'qa': {'qa': 'qa_per_video.csv'},
    'events': {'motion': 'motion_per_video.csv', 'timing': 'timing_per_video.csv'},
    'quality': {'quality': 'quality/quality_per_video.csv'},
    'geometry': {'geometry': 'geometry_per_video.csv'},
    'camera': {'camera': 'camera/video_metrics.csv'},
}


def rows(group, row):
    """Yield (metric, value, status) for one native CSV row of a metric group."""
    if group == 'qa':
        for k in QA_KEYS:
            yield k, row.get(k), row.get(k + '_status')
    elif group == 'motion':
        yield 'ISR', row.get('ISR'), row.get('status')
    elif group in ('timing', 'timing_repair'):
        if row.get('metric', 'EventTimingCoverage') == 'EventTimingCoverage':
            yield 'EventTimingCoverage', row.get('value'), row.get('status')
    elif group == 'quality':
        for k in QUALITY:
            yield k, row.get(k), row.get('status')
    elif group == 'geometry':
        if 'metric' in row:
            if row['metric'] in GEOMETRY:
                yield row['metric'], row.get('value'), row.get('status')
        else:
            for k in GEOMETRY:
                yield k, row.get(k), row.get('status')
    elif group == 'camera':
        if model_config.is_camera_reference(row['model'], row['cell']):
            return  # the reference trajectory itself
        for k in CAMERA:
            yield k, row.get(k), row.get('status')


def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def long_rows(group, path, keep=lambda case: True):
    for row in read(path):
        if not keep(row['case']):
            continue
        for metric, value, status in rows(group, row):
            complete = status == 'complete' and value not in ('', None)
            number = float(value) if complete else None
            if complete and not math.isfinite(number):
                raise ValueError(f'non-finite value {row["model"]}/{row["case"]}/{row["cell"]}/{metric} in {path}')
            yield dict(model=row['model'], case=row['case'], cell=row['cell'], metric=metric,
                       value=repr(number) if complete else '', status=status if complete else (status or 'missing'))


def write(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def export(out, stage):
    """Write <out>/<stage>/per_video.csv from the stage's native outputs."""
    out = Path(out)
    records = [r for group, rel in SOURCES[stage].items() if (out / rel).is_file() for r in long_rows(group, out / rel)]
    write(out / stage / 'per_video.csv', records)
    return out / stage / 'per_video.csv'
