from pathlib import Path

import yaml

from worlddraft_eval.geometry import run as geometry_eval
from worlddraft_eval.common import models
from worlddraft_eval.quality import run as quality_eval
from worlddraft_eval.common import judge as vllm_engine
from worlddraft_eval.pipeline import config, launch

ROOT = config.ROOT


def use_config(monkeypatch, tmp_path, base, local=None):
    (tmp_path / 'envs.yaml').write_text(yaml.safe_dump(base))
    if local is not None:
        (tmp_path / 'envs.local.yaml').write_text(yaml.safe_dump(local))
    monkeypatch.setattr(config, 'ENVS', tmp_path / 'envs.yaml')
    monkeypatch.setattr(config, 'LOCAL', tmp_path / 'envs.local.yaml')
    config.envs.cache_clear()


def test_models_are_named_by_hugging_face_id():
    assert vllm_engine.MODEL == 'Qwen/Qwen3.6-27B'
    assert geometry_eval.TRACKERS == {'sam3': 'facebook/sam3', 'sam2': 'facebook/sam2.1-hiera-large', 'identity': 'facebook/dinov2-base'}
    assert launch.CV_UTILS == ('kernels-community/cv-utils', '50fb2dfddda16b050e60328925b4f261d946a617')


def test_configured_local_directory_wins_over_download(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path, {'python': {}}, {'models': {'facebook/sam3': str(tmp_path / 'sam3'), 'Qwen/Qwen3.6-27B': 'weights/qwen'}})
    monkeypatch.setattr(models, 'snapshot_download', lambda *a, **k: (_ for _ in ()).throw(AssertionError('no download')))
    assert models.local('facebook/sam3') == tmp_path / 'sam3'
    assert models.local('Qwen/Qwen3.6-27B') == ROOT / 'weights/qwen'
    config.envs.cache_clear()


def test_unconfigured_model_downloads_the_pinned_revision(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path, {'python': {}})
    calls = []
    monkeypatch.setattr(models, 'snapshot_download', lambda repo_id, revision=None: calls.append((repo_id, revision)) or str(tmp_path))
    assert models.local(*launch.CV_UTILS) == tmp_path
    assert calls == [launch.CV_UTILS]
    config.envs.cache_clear()


def test_quality_weights_default_to_user_cache(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path, {'python': {}})
    assert models.quality_weights() == Path('~/.cache/worlddraft/vbench-quality').expanduser()
    use_config(monkeypatch, tmp_path, {'python': {}}, {'quality_weights': '/w/q'})
    assert models.quality_weights() == Path('/w/q')
    config.envs.cache_clear()


def test_local_overrides_merge_per_key(tmp_path, monkeypatch):
    use_config(monkeypatch, tmp_path, {'python': {'qa': 'python', 'camera': 'python'}, 'gpus': [0]},
               {'python': {'camera': '/opt/vipe/bin/python'}, 'gpus': [4, 5], 'vipe_root': 'third/vipe'})
    assert config.python('qa') == 'python' and config.python('camera') == '/opt/vipe/bin/python'
    assert config.envs()['gpus'] == [4, 5]
    assert config.envs()['vipe_root'] == str(ROOT / 'third/vipe')
    config.envs.cache_clear()


def test_launcher_lists_the_evaluation_modules():
    assert {'qa.run', 'events.run', 'geometry.run', 'geometry.tracks'} <= set(launch.MODULES)
    assert quality_eval.models is models
