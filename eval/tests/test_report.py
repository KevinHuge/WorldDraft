from worlddraft_eval.report import tables as T
from worlddraft_eval.report.tables import CELLS, common_color_ids

def test_same_complete_cohort_and_delta_sign(monkeypatch):
    monkeypatch.setattr(T,'CASES',['A01','A02'])
    v={('m',c,cell,'RF_total'):x for c in ['A01','A02'] for cell,x in [('P0_AN',.2),('P1_AN',.5),('P2_AN',.4)]}
    v.update({('n',c,cell,'RF_total'):x for c in ['A01','A02'] for cell,x in [('P0_AN',.1),('P1_AN',.3),('P2_AN',.2)]})
    del v['n','A02','P2_AN','RF_total']
    values,n=T.summarize(v,['m','n'],'RF_total',['P0_AN','P1_AN','P2_AN'],('P1_AN','P2_AN'))
    assert n==1
    assert round(values['m'],6)==-10 and round(values['n'],6)==-10

def test_empty_is_missing_not_zero(monkeypatch):
    monkeypatch.setattr(T,'CASES',['A01'])
    v,n=T.summarize({},['m'],'bbox_iou',['P1_AN'])
    assert v=={'m':None} and n==0

def test_camera_keeps_units_and_case_weights(monkeypatch):
    monkeypatch.setattr(T,'CASES',['A01','A02'])
    v={('m',c,cell,'RotErr_deg_mean'):x for c,x in [('A01',2),('A02',6)] for cell in ['P0_AN','P1_AN']}
    s,n=T.summarize(v,['m'],'RotErr_deg_mean',['P0_AN','P1_AN'])
    assert s['m']==4 and n==2


def test_cohort_is_set_by_generators_and_draft_is_scored_on_it(monkeypatch):
    monkeypatch.setattr(T,'CASES',['A01','A02'])
    v={(m,c,'P1_AN','RF_total'):x for m,x in [('g1',.2),('g2',.4)] for c in ['A01','A02']}
    v['draft','A01','P1_AN','RF_total']=.9  # draft incomplete on A02: shown as missing, families unchanged
    values,n=T.summarize(v,['g1','g2'],'RF_total',['P1_AN'],reference='draft')
    assert n==2 and round(values['g1'],6)==20 and values['draft'] is None
    v['draft','A02','P1_AN','RF_total']=.7
    values,n=T.summarize(v,['g1','g2'],'RF_total',['P1_AN'],reference='draft')
    assert n==2 and round(values['draft'],6)==80


def test_generators_come_from_the_results_in_config_order():
    values={(m,'A01','P1_AN','RF_total'):1 for m in ['Wan-3.0','draft','MiniMax','C2R']}
    assert T.generators(values)==['MiniMax','Wan-3.0','C2R']


def test_all_models_use_same_complete_families():
    from worlddraft_eval.report.all_models import summary
    values={(model,case,cell,'ISR'):score
            for model,score in [('a',.2),('b',.8),('draft',1.)]
            for case in ['A01','A02'] for cell in CELLS}
    del values['b','A02','P1_AN','ISR']
    rows,coverage=summary(values,['a','b'],['ISR','bbox_iou'])
    assert [row['ISR'] for row in rows]==[20.,80.,100.]
    assert all(row['bbox_iou'] is None for row in rows)
    assert coverage[0]['cases']=='A01'
    assert coverage[0]['videos_per_model']==9


def test_common_conflict_matches_part_target_and_all_polish_levels():
    def q(pid, metric, expected, entity='car'):
        return dict(item_id='C01', proposition_id=pid, metric=metric, attribute='color',
                    expected_answer=expected, entity_id=entity)
    questions = [q('1', 'RF', 'yes'), q('2', 'RF', 'no'), q('3', 'SL', 'no'),
                 q('4', 'RF', 'yes'), q('5', 'RF', 'no'), q('6', 'SL', 'no'),
                 q('7', 'RF', 'yes'), q('8', 'RF', 'no')]
    metadata = {'1': dict(subject='body', target='green'), '2': dict(paired_with='1'),
                '3': dict(subject='body', target='green'),
                '4': dict(subject='roof', target='blue'), '5': dict(paired_with='4'),
                '6': dict(subject='roof', target='blue'),
                '7': dict(subject='wheels', target='green'), '8': dict(paired_with='7')}
    result = common_color_ids(questions, metadata, {'6': {'cells': ['P1_AC', 'P2_AC']}})
    assert result == {'1': dict(negative='2', source_ids=['3'])}


def write_long(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('model,case,cell,metric,value,status\n' + ''.join(','.join(map(str, r)) + '\n' for r in rows))


def test_collect_reads_complete_rows_from_every_result_dir(tmp_path):
    from worlddraft_eval.report import results as published
    write_long(tmp_path/'a/qa/per_video.csv', [('M', 'A01', 'P1_AN', 'RF_total', repr(0.1+0.2), 'complete'),
                                               ('M', 'A01', 'P1_AN', 'AP_total', '', 'pending')])
    write_long(tmp_path/'b/events/per_video.csv', [('N', 'A01', 'P1_AN', 'ISR', '1.0', 'complete')])
    assert published.collect([tmp_path/'a', tmp_path/'b']) == {('M', 'A01', 'P1_AN', 'RF_total'): 0.1+0.2,
                                                                ('N', 'A01', 'P1_AN', 'ISR'): 1.0}


def test_collect_rejects_the_same_score_in_two_result_dirs(tmp_path):
    import pytest
    from worlddraft_eval.report import results as published
    for d in ('a', 'b'):
        write_long(tmp_path/d/'qa/per_video.csv', [('M', 'A01', 'P1_AN', 'RF_total', '0.5', 'complete')])
    with pytest.raises(ValueError, match='appears in both'):
        published.collect([tmp_path/'a', tmp_path/'b'])
