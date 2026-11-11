"""Model weights by Hugging Face id: a directory configured under `models:` in eval/configs/envs(.local).yaml, or a
download into the Hugging Face cache on first use."""
from pathlib import Path

from huggingface_hub import snapshot_download

from worlddraft_eval.pipeline.config import ROOT, envs

QUALITY_WEIGHTS = '~/.cache/worlddraft/vbench-quality'  # filled by eval/download_quality_weights.sh


def _path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def local(model_id, revision=None):
    """Local directory holding the weights of `model_id`."""
    configured = (envs().get('models') or {}).get(model_id)
    if configured:
        return _path(configured)
    return Path(snapshot_download(model_id, revision=revision))


def quality_weights():
    """MUSIQ / CLIP / LAION aesthetic weights (not on the Hugging Face hub)."""
    return _path(envs().get('quality_weights') or QUALITY_WEIGHTS)
