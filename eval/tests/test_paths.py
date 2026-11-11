from worlddraft_eval.gt import visible as gt
from worlddraft_eval.common import paths
from worlddraft_eval.common.io import repo_path


def test_data_records_resolve_into_the_data_directory():
    assert repo_path('data/gt/A01/gt.json') == paths.GT / 'A01/gt.json'
    assert repo_path('data/drafts/A01/A01_P2_AA/preprocessing.json') == paths.preprocessing('A01')


def test_repo_relative_records_resolve_against_the_repo():
    assert repo_path('eval/protocol/contracts/A01.json') == paths.PROTOCOL / 'contracts/A01.json'


def test_paths_recorded_on_other_machines_map_to_the_local_layout():
    assert repo_path('/elsewhere/draft/data/gt/A01/gt.json') == paths.GT / 'A01/gt.json'
    assert repo_path('/elsewhere/draft/data/scenes/A01/A01_P2_AA.blend') == paths.scene_blend('A01')
    assert repo_path('/elsewhere/draft/eval/protocol/contracts/A01.json') == paths.PROTOCOL / 'contracts/A01.json'


def test_existing_absolute_paths_are_kept(tmp_path):
    assert repo_path(tmp_path) == tmp_path


def test_gt_records_data_paths_with_the_data_prefix(tmp_path):
    assert gt.repo_rel(paths.preprocessing('A01')) == 'data/drafts/A01/A01_P2_AA/preprocessing.json'
    assert gt.repo_rel(paths.PROTOCOL / 'contracts/A01.json') == 'eval/protocol/contracts/A01.json'
    assert gt.repo_rel(tmp_path / 'x.png') == str(tmp_path / 'x.png')
