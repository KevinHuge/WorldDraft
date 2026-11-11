"""QA engine: evidence, gating, scoring and the benchmark-wide live snapshot."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import pytest

from worlddraft_eval.common import paths
from worlddraft_eval.qa import engine as D
from worlddraft_eval.qa.evidence import output_mask_gt
from worlddraft_eval.qa.run import snapshot_rows


def question(pid,metric='RF',attribute='color',eid='car'):
    return dict(zip(D.FIELDS,['A01','',str(pid),'0','entity' if metric=='EntityCount' else 'attribute',
        'count' if metric=='EntityCount' else attribute,'fact','How many cars are in the full image?' if metric=='EntityCount' else "Is this car's body green?",
        attribute,eid,metric,'' if metric=='EntityCount' else ('no' if metric=='SL' else 'yes'),'AC' if metric=='SL' else 'ALL']))


class QAEngineTest(unittest.TestCase):
    def test_rf_selects_one_reference_frame(self):
        frames={t:dict(boxes={'car':dict(visible=v,visible_mask_area=a)})
                for t,v,a in [(0,True,10),(1,True,20),(2,True,20),(3,False,100)]}
        self.assertEqual(D.rf_time([3,2,1,0],frames,['car']),1)
        self.assertEqual(D.rf_time([3,2,1,0],frames,[]),0)
        self.assertEqual(D.rf_time([3],frames,['car']),3)

    def test_counts_use_visibility_and_independent_regions(self):
        f={'boxes':{'a':dict(visible=True,box=[0,0,10,10],center=[5,5]),
                    'b':dict(visible=True,box=[3,3,12,12],center=[7,7]),
                    'c':dict(visible=False,box=None,center=None)}}
        self.assertEqual(D.count_gt(f,['a','b','c'],'global',''),2)
        self.assertEqual(D.count_gt(f,['a','b','c'],'region','a'),2)
        self.assertEqual(D.count_gt(f,['a','b','c'],'region','b'),2)

    def test_question_weighting_binary_errors_and_sl_direction(self):
        rows=[dict(question(1),eligible=True,expected='yes',raw_answer=x) for x in ['yes','no','no']]
        rows += [dict(question(2),eligible=True,expected='yes',raw_answer='yes')]
        self.assertAlmostEqual(D.aggregate(rows,'RF')['value'],2/3)
        self.assertEqual(D.aggregate(rows,'RF','texture')['status'],'not_applicable')
        rows[0]['raw_answer']='unknown'
        self.assertIsNone(D.aggregate(rows,'RF')['value'])
        sl=[dict(question(3,'SL'),eligible=True,raw_answer=x) for x in ['yes','no']]
        self.assertEqual(D.aggregate(sl,'SL')['value'],.5)
        self.assertEqual(D.aggregate(sl,'SL')['n_scored'],2)
        count=dict(question(4,'EntityCount',''),eligible=True,raw_answer='2',expected=2)
        self.assertEqual(D.parse(count),(2,'answered'))
        self.assertEqual(D.row_score(count),1)
        count['expected']=3
        self.assertEqual(D.row_score(count),0)
        count['raw_answer']='-1'
        self.assertEqual(D.parse(count)[1],'parse_error')

    def test_binary_scores_match_expected_answers_without_crediting_invalids(self):
        for metric in ('RF','AP'):
            for expected in ('yes','no'):
                row=dict(question(1,metric),eligible=True,expected=expected)
                for answer in ('yes','no'):
                    with self.subTest(metric=metric,expected=expected,answer=answer):
                        row['raw_answer']=answer
                        self.assertEqual(D.row_score(row),int(answer==expected))
                for raw in (None,'unknown','invalid',''):
                    row['raw_answer']=raw
                    self.assertIsNone(D.row_score(row))
                row.update(eligible=False,raw_answer=expected)
                self.assertIsNone(D.row_score(row))

    def test_question_loader_rejects_nonbinary_expectations_and_positive_sl(self):
        config=dict(episode='A01',qa=dict(groups={'A01':{}}))
        gt=dict(entities={'car':{}})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'questions.csv'
            for metric,expected in [('RF','unknown'),('AP',''),('SL','yes')]:
                with self.subTest(metric=metric,expected=expected):
                    with path.open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=D.FIELDS)
                        writer.writeheader()
                        writer.writerow(dict(question(1,metric),expected_answer=expected))
                    with self.assertRaisesRegex(ValueError,'expected answer'):
                        D.load_questions(path,config,gt)

    def test_count_scopes_weighting_and_incomplete_answers(self):
        rows=[dict(question(1,'EntityCount',''),count_scope='global',eligible=True,expected=8,raw_answer=x)
              for x in ['8','7','9']]
        rows += [dict(question(2,'EntityCount',''),count_scope='region',eligible=True,expected=1,raw_answer='1')]
        self.assertEqual(D.aggregate(rows,'EntityCount',count_scope='global')['value'],0)
        rows[1]['raw_answer']='8'; rows[2]['raw_answer']='8'
        self.assertEqual(D.aggregate(rows,'EntityCount',count_scope='global')['value'],1)
        self.assertEqual(D.aggregate(rows,'EntityCount',count_scope='region')['value'],1)
        rows[0]['raw_answer']=None
        self.assertIsNone(D.aggregate(rows,'EntityCount',count_scope='global')['value'])
        rows[0].update(eligible=False)
        self.assertEqual(D.aggregate(rows,'EntityCount',count_scope='global')['value'],1)

    def test_prepare_judge_resume_and_evidence_integrity(self):
        import av
        class Judge:
            metadata={'backend':'synthetic_test'}
            calls=0
            def run(self,contents):
                self.calls+=1
                return ['1' if 'How many' in c[-1]['text'] else 'no' if 'magenta' in c[-1]['text'] else 'yes' for c in contents]
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);(base/'videos/P0_AC').mkdir(parents=True)
            v=base/'videos/P0_AC/output.mp4'
            with av.open(str(v),'w') as container:
                stream=container.add_stream('mpeg4',rate=16);stream.width=64;stream.height=48;stream.pix_fmt='yuv420p'
                for _ in range(3):
                    frame=av.VideoFrame.from_ndarray(np.zeros((48,64,3),np.uint8),format='rgb24')
                    for packet in stream.encode(frame):container.mux(packet)
                for packet in stream.encode():container.mux(packet)
            sparse=dict(source_size=[64,48],source_blend='test.blend',source_cell='A01_P2_AA',frames=[dict(model_input_index=i,model_input_time_s=i/16,source_index=i,blender_frame=i+1,entities=[dict(entity_id='car',visible=True,mask_area=100,bbox_xyxy=[.25,.25,.5,.5],center_xy=[.375,.375])]) for i in range(3)])
            dense=dict(episode='A01',camera='blender',width=64,height=48,fps=16,frame_count=3,entities={'car':dict(label='car')})
            D.write_json(base/'sparse.json',sparse);D.write_json(base/'dense.json',dense)
            (base/'prompt.txt').write_text('a green car');D.write_json(base/'appearance.json',{})
            qs=[question(1),question(2,'AP'),question(3,'SL'),question(4,'EntityCount','','cars'),question(5,'EntityCount','')]
            qs += [dict(question(6,'RF'),expected_answer='no',question_natural_language="Is this car's body magenta?"),
                   dict(question(7,'AP'),expected_answer='no',question_natural_language="Is this car's body magenta throughout the sequence?")]
            qs += [dict(question(8,'AP')), dict(qs[6],proposition_id='9')]
            qs[1]['dependency']='8';qs[6]['dependency']='9'
            with (base/'questions.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=D.FIELDS);w.writeheader();w.writerows(qs)
            config=dict(protocol='test',episode='A01',files=dict(questions='questions.csv',visibility_gt='sparse.json',entities_gt='dense.json',target_prompt='prompt.txt',appearance='appearance.json'),qa=dict(groups={'A01':{'cars':['car']}},bindings={'A01':{'4':{'count_scope':'global'},'5':{'count_scope':'region'}}},times_s=[0,1/16,2/16],count_times_s=[0,2/16],crop='gt_bbox',crop_side=224))
            D.write_json(base/'settings.json',config)
            out=base/'results';run=D.prepare(base/'settings.json',base/'videos','test',['P0_AC'],out,'test')
            rows=D.read_json(run/'P0_AC/answers.json')
            rf=[r for r in rows if r['metric']=='RF']
            self.assertEqual(len(rf),2)
            self.assertEqual(len(rf[0]['frames']),1)
            self.assertEqual(rf[0]['frames'][0]['gt_time'],0)
            ap=next(r for r in rows if r['metric']=='AP')
            self.assertEqual(len(ap['images']),6)
            anchor=next(r for r in rows if r['proposition_id']=='8')
            self.assertEqual(len(anchor['frames']),1)
            self.assertEqual(anchor['frames'][0]['gt_time'],0)
            self.assertEqual(ap['frames'][0]['regions']['car']['video_box'],[16,12,32,24])
            self.assertEqual(ap['frames'][0]['regions']['car']['crop_box'],[16,12,32,24])
            self.assertEqual(ap['prompt'],ap['question_natural_language']+'\nAnswer yes or no.')
            self.assertNotIn('1',next(r for r in rows if r['metric']=='EntityCount')['prompt'])
            judge=Judge();D.judge(out,'test',engine=judge);calls=judge.calls
            D.judge(out,'test',engine=judge);self.assertEqual(judge.calls,calls)
            videos=D.report(out,'test')
            self.assertEqual(videos[0]['RF_total'],1)
            self.assertEqual(videos[0]['AP_total'],1)
            self.assertEqual(videos[0]['SL_color'],1)
            self.assertEqual(videos[0]['n_pending'],0)
            self.assertEqual(videos[0]['DF_count_global_accuracy'],1)
            self.assertEqual(videos[0]['DF_count_local_accuracy'],1)
            self.assertIsNone(videos[0]['DF_total'])
            with (out/'summary.csv').open(encoding='utf-8-sig',newline='') as f:
                summary={r['metric']:r for r in csv.DictReader(f)}
            self.assertEqual(summary['DF_count_global_accuracy']['value'],'1.0')
            self.assertEqual(summary['DF_total']['status'],'not_computed')
            rows=D.read_json(run/'P0_AC/answers.json');rows[0]['raw_answer']=None
            D.write_json(run/'P0_AC/answers.json',rows)
            (run/rows[0]['images'][0]['path']).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError,'Evidence changed'):D.judge(out,'test',engine=judge)
            qs[0]['attribute']='typo'
            qs += [dict(question(8,'AP')), dict(qs[6],proposition_id='9')]
            qs[1]['dependency']='8';qs[6]['dependency']='9'
            with (base/'questions.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=D.FIELDS);w.writeheader();w.writerows(qs)
            with self.assertRaisesRegex(ValueError,'attribute'):D.load_questions(base/'questions.csv',config,dense)


class EvidenceTest(unittest.TestCase):
    def test_live_snapshot_matches_report_without_writing_group_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp);(directory/'P0_AC').mkdir()
            rows=[]
            for i,(metric,scope,raw,eligible) in enumerate([
                    ('EntityCount','global','2',True),('EntityCount','region','1',True),
                    ('RF',None,'yes',True),('RF',None,None,True),
                    ('AP',None,'no',True),('SL',None,'yes',True),('RF',None,None,False)]):
                rows.append(dict(model='test',item_id='A01',cell='P0_AC',proposition_id=str(i+1),
                    sample_id='sample',metric=metric,attribute='' if metric=='EntityCount' else 'color',
                    entity_id='car',count_scope=scope,question_natural_language='Test question?',
                    expected=2 if scope=='global' else 1 if scope=='region' else 'yes',
                    raw_answer=raw,eligible=eligible,images=[]))
            D.write_json(directory/'P0_AC/answers.json',rows)
            D.write_json(directory/'plan.json',dict(model='test',episode='A01',protocol='test',
                inputs={},score_rules={},jobs=[dict(path='P0_AC',cell='P0_AC')]))
            expected=D.report(directory,None)
            before={str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()}
            self.assertEqual(snapshot_rows(directory),expected)
            self.assertEqual(before,{str(p):p.read_bytes() for p in directory.rglob('*') if p.is_file()})
            self.assertIsNone(expected[0]['RF_total'])
            self.assertEqual(expected[0]['n_pending'],1)

    def test_visible_count_and_bbox_follow_generator_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            masks=Path(tmp)
            labels=np.zeros((4,8),dtype=np.uint16)
            labels[:,:2]=1;labels[:,3:7]=2
            Image.fromarray(labels).save(masks/'mask_000.png')
            sparse=dict(source_size=[8,4],letterbox=dict(resized_size=[8,4],offset_xy=[0,0]),
                        entities=[dict(entity_id='a',instance_id=1),dict(entity_id='b',instance_id=2)])
            canonical=dict(width=8,height=4,adaptation={},frames=[dict(frame=0,boxes={})])
            gt=output_mask_gt(sparse,canonical,'OmniWeaving',(4,4),masks)
            frame=gt['frames'][0]
            self.assertFalse(frame['boxes']['a']['visible'])
            self.assertEqual(frame['boxes']['b']['box'],[1,0,4,4])
            self.assertEqual(frame['boxes']['b']['center'],[2.5,2.0])
            self.assertEqual(D.count_gt(frame,['a','b'],'global',''),1)
            self.assertEqual(canonical['frames'][0]['boxes'],{})

    def test_sparse_entity_labels_are_explicit(self):
        sparse=dict(source_size=[8,4],source_blend='test',source_cell='A01_P2_AA',frames=[dict(
            model_input_index=0,model_input_time_s=0,source_index=0,blender_frame=1,
            entities=[dict(entity_id='arm_part',visible=True,mask_area=4,
                           bbox_xyxy=[0,0,.5,.5],center_xy=[.25,.25])])])
        dense=dict(width=8,height=4,episode='A01',camera='camera',fps=16,frame_count=1,
                   entities={'whole_robot':dict(label='robot')})
        with self.assertRaisesRegex(ValueError,'entity sets differ'):D.visible_gt(sparse,dense)
        gt=D.visible_gt(sparse,dense,{'arm_part':'arm segment'})
        self.assertEqual(gt['entities'],{'arm_part':dict(label='arm segment')})
        with self.assertRaisesRegex(ValueError,'every visible-mask'):D.visible_gt(sparse,dense,{'wrong':'arm'})


def pair(expected='yes', first='yes', second='yes'):
    base=dict(metric='AP',attribute='color',entity_id='car',expected=expected,
              eligible=True,count_scope='',dependency='0')
    return [dict(base,proposition_id='1',raw_answer=first),
            dict(base,proposition_id='2',dependency='1',raw_answer=second)]


@pytest.mark.parametrize('expected,first,second,value',[
    ('yes','yes','yes',1),('yes','yes','no',0),('yes','no','yes',0),
    ('yes','no','no',0),('no','no','no',1),('no','yes','no',0),
    ('no','no','yes',0),('no','yes','yes',0),('yes','no',None,0)])
def test_gating_and_no_presence_half_credit(expected,first,second,value):
    rows=pair(expected,first,second)
    result=D.aggregate(rows,'AP')
    assert result['value']==value
    assert result['n_expected']==1
    assert D.gated_row_score(rows[1],rows)==value


@pytest.mark.parametrize('first,status',[(None,'pending'),('bad','parse_error')])
def test_unresolved_parent_does_not_score(first,status):
    rows=pair(first=first)
    assert D.gated_row_score(rows[1],rows) is None
    assert D.aggregate(rows,'AP')['status']==status
    assert D.aggregate(rows,'AP')['value'] is None


def test_invisible_parent_does_not_score():
    rows=pair();rows[0]['eligible']=False
    assert D.gated_row_score(rows[1],rows) is None
    assert D.aggregate(rows,'AP')['value'] is None


@pytest.mark.parametrize('field,value',[('entity_id','truck'),('attribute','material'),
                                       ('expected','no'),('dependency','2')])
def test_bad_runtime_parent_rejected(field,value):
    rows=pair();rows[0][field]=value
    with pytest.raises(ValueError):D.gated_row_score(rows[1],rows)


@pytest.mark.parametrize('field,value',[('dependency','999999'),('dependency','2'),
                                        ('attribute','material'),('expected_answer','no'),
                                        ('entity_id','other'),('tuple','different target')])
def test_loader_rejects_invalid_binding(tmp_path,field,value):
    base=dict(item_id='T',keywords='car',category_broad='attribute',category_detailed='color',
              tuple='color(car,green)',question_natural_language='Is this car green?',
              attribute='color',entity_id='car',metric='AP',expected_answer='yes',applies_to='ALL')
    rows=[dict(base,proposition_id='1',dependency='0'),dict(base,proposition_id='2',dependency='1')]
    rows[1][field]=value
    path=tmp_path/'questions.csv'
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=D.FIELDS);writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError):
        D.load_questions(path,dict(episode='T',qa=dict(groups={'T':{}})),dict(entities={'car':{},'other':{}}))


if __name__=='__main__':unittest.main()


def test_repo_path_maps_paths_recorded_on_another_machine():
    from worlddraft_eval.common.io import ROOT, repo_path
    assert repo_path('/elsewhere/draft/data/gt/A01/gt.json') == paths.GT/'A01/gt.json'
    assert repo_path('/elsewhere/x.png') == Path('/elsewhere/x.png')
