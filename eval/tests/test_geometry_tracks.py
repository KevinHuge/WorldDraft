from worlddraft_eval.geometry.tracks import distance_status, track_summary


def test_tracks_exclude_static_candidates_and_keep_missing_distance_na():
    gt = dict(entities={'moving': dict(dynamic=True, label='ball'),
                        'static': dict(dynamic=False, interaction_candidate=True, label='basin')})
    entity = dict(per_entity={
        'moving': dict(n_visible_instants=2, mean_center_err=.2, miss_rate=.5,
                       instants=[dict(gt=[0], pred=[0]), dict(gt=[0], pred=None)]),
        'static': dict(n_visible_instants=1, mean_center_err=.8, miss_rate=0,
                      instants=[dict(gt=[0], pred=[0])])})
    result = track_summary(entity, gt)
    assert result['track_ade'] == .2
    assert result['track_miss_rate'] == result['track_matched_coverage'] == .5
    assert result['n_dynamic_matched_instants'] == 1
    assert result['track_ade_alias_of'] == ''
    entity['per_entity']['moving'].update(mean_center_err=None, miss_rate=1,
                                         instants=[dict(gt=[0], pred=None)] * 2)
    result = track_summary(entity, gt)
    assert result['track_ade'] is None
    assert result['track_matched_coverage'] == 0
    assert result['track_miss_rate'] == 1


def test_missing_predictions_remain_applicable_without_a_fake_distance():
    assert distance_status(None, 1, 0) == 'no_matched_predictions'
    assert distance_status(None, 0, 0) == 'not_applicable'
    assert distance_status(0.0, 1, 1) == 'complete'


def test_export_keeps_canonical_csv_until_all_ten_metrics_exist(tmp_path, monkeypatch):
    import csv
    from worlddraft_eval.geometry import tracks as T
    out = tmp_path / 'geometry_per_video.csv'
    out.write_text('previous complete snapshot')

    def base_export(inventory, runs, destination, verify_inputs):
        assert destination != out
        assert out.read_text() == 'previous complete snapshot'
        base = dict(model='m', case='A01', cell='P0_AN', status='missing')
        T.write_csv(tmp_path / 'geometry_video_details.csv', [base])
        metrics = ['bbox_iou', 'position_error', 'track_ade', 'miss_rate', 'matched_coverage',
                   'dinov2_appearance_short', 'dinov2_appearance_revisit', 'dinov2_appearance_cross']
        T.write_csv(destination, [dict(base, metric=m, value=None) for m in metrics])
        (tmp_path / 'geometry_status.json').write_text('{}')

    monkeypatch.setattr(T, 'export', base_export)
    T.export_tracks('unused', [], out)
    with out.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10
    assert {r['metric'] for r in rows} >= {'track_ade', 'track_miss_rate', 'track_matched_coverage'}
    assert not (tmp_path / '.geometry_base_export.csv').exists()
