import pytest

from worlddraft_eval.geometry.export import geometry_summary, input_crop


def test_actual_generation_crop():
    assert input_crop('UniVideo', (832, 480), (832, 480)) == [0, 0, 832, 480]
    assert input_crop('VINO', (832, 480), (832, 480)) == pytest.approx([0, 0, 832 * 832 / 840, 480 * 480 / 485])
    assert input_crop('OmniWeaving', (832, 480), (800, 480)) == [16, 0, 800, 480]
    assert input_crop('UniVideo', (480, 832), (480, 832)) == [0, 0, 480, 832]
    assert input_crop('VINO', (480, 832), (480, 832)) == pytest.approx([0, 0, 480 * 480 / 485, 832 * 832 / 840])
    with pytest.raises(ValueError):
        input_crop('VINO', (832, 480), (848, 480))


def test_coverage_preserves_entity_weighting_and_distance_na():
    result = geometry_summary(dict(summary=dict(mean_iou=.3, mean_center_err=.2, miss_rate=.5,
        n_entities=2, n_tracks=1, n_assigned=1), per_entity={
            'a': dict(n_visible_instants=1, instants=[dict(gt=[0], pred=[0])]),
            'b': dict(n_visible_instants=3, instants=[dict(gt=[0], pred=None)] * 3)}))
    assert result['matched_coverage'] == .5
    assert result['matched_instant_coverage'] == .25
    assert result['position_error'] == result['track_ade'] == .2
    empty = geometry_summary(dict(summary=dict(mean_iou=0, mean_center_err=None, miss_rate=1,
        n_entities=1, n_tracks=0, n_assigned=0), per_entity={
            'a': dict(n_visible_instants=1, instants=[dict(gt=[0], pred=None)])}))
    assert empty['track_ade'] is None
    assert empty['matched_instant_coverage'] == 0


def test_whole_canvas_models_and_unverified_resize():
    assert input_crop('C2R', [832, 480], [832, 480]) == [0, 0, 832, 480]
    assert input_crop('Wan-3.0', [832, 480], [1920, 1080]) == [0, 0, 832, 480]
    with pytest.raises(ValueError):
        input_crop('C2R', [832, 480], [480, 832])
