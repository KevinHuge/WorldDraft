import json
import unittest
from worlddraft_eval.gt.events import timestamp_interval, boundaries
from worlddraft_eval.common import paths
from worlddraft_eval.events.run import parse_timing, timing_score, event_score, motion_requests, canonical_size, load


class EventTest(unittest.TestCase):
    def test_negative_carton_questions_preserve_the_selected_target(self):
        from worlddraft_eval.events.run import NEGATIVE_QUESTIONS
        cases=load(NEGATIVE_QUESTIONS)['cases']
        self.assertIn('same leading carton nearest the downstream end',cases['B08']['LRA']['final_carton_stop'])
        for fact in cases['F06']['ISR']['pickup'].values():
            self.assertIn('selected first arriving carton',fact)

    def test_factual_contrasts_pair_every_scored_binary_answer(self):
        from collections import Counter
        from worlddraft_eval.events.run import NEGATIVE_QUESTIONS, binary_prompt
        from worlddraft_eval.gt.events import PROTOCOL
        protocol=load(PROTOCOL/'motion_protocol.json');contrasts=load(NEGATIVE_QUESTIONS)['cases']
        self.assertEqual(set(contrasts),set(protocol['cases']))
        counts={kind:Counter() for kind in ('ISR','LRA','EndState')}
        for case,spec in protocol['cases'].items():
            legacy=motion_requests(spec,protocol['settings'])
            rows=motion_requests(spec,protocol['settings'],contrasts[case])
            self.assertEqual(len(rows),2*len(legacy))
            self.assertEqual(len({r['id'] for r in rows}),len(rows))
            for old,positive,negative in zip(legacy,rows[::2],rows[1::2],strict=True):
                with self.subTest(case=case,question=positive['id']):
                    self.assertEqual(positive['pair_id'],negative['pair_id'])
                    self.assertEqual(positive['id'],old['id'])
                    for key,value in old.items():self.assertEqual(positive[key],value)
                    self.assertEqual(positive['window'],negative['window'])
                    self.assertEqual(positive['participants'],negative['participants'])
                    old['frames']=positive['frames']=negative['frames']=[]
                    self.assertEqual(binary_prompt(old),binary_prompt(positive))
                    self.assertNotIn('"expected":',binary_prompt(negative))
                    if positive['kind']=='EndState' and positive.get('target_context'):
                        self.assertNotIn(positive['target_context'],binary_prompt(positive))
                        self.assertNotIn(positive['target_context'],binary_prompt(negative))
                    if positive['kind']=='ISR':
                        self.assertEqual(negative['expected']['participants'],'yes')
                        text=' '.join(negative['data'][k] for k in ('action','outcome'))
                    elif positive['kind']=='LRA':text=negative['data']['text']
                    else:text=' '.join(negative['facts'])
                    self.assertIn('It is not the case that:',text)
                    if positive['kind']=='ISR':
                        for key in ('action','outcome'):
                            self.assertEqual(negative['data'][key],'It is not the case that: '+positive['data'][key])
                    elif positive['kind']=='LRA':
                        self.assertEqual(negative['data']['text'],'It is not the case that: '+positive['data']['text'])
                    else:
                        self.assertEqual(negative['facts'],['It is not the case that: '+f for f in positive['facts']])
                    for row in (positive,negative):
                        counts[row['kind']].update(value for key,value in row['expected'].items() if key!='participants')
        for kind,count in counts.items():
            with self.subTest(kind=kind):
                self.assertGreater(count['no'],0)
                self.assertEqual(count['yes'],count['no'])
        self.assertNotIn('tray',json.dumps(contrasts['A08']))

    def test_contrast_coverage_must_match_the_current_protocol(self):
        spec=dict(events=[],timeline=[],groups={},end_state=['A ball is visible.'])
        with self.assertRaisesRegex(ValueError,'do not match'):
            motion_requests(spec,{},dict(ISR={},LRA={},EndState=[]))

    def test_expected_answers_score_both_polarities_and_all_end_state_rows(self):
        from worlddraft_eval.events.run import NEGATIVE_QUESTIONS
        from worlddraft_eval.gt.events import PROTOCOL
        protocol=load(PROTOCOL/'motion_protocol.json')
        rows=motion_requests(protocol['cases']['A01'],protocol['settings'],load(NEGATIVE_QUESTIONS)['cases']['A01'])
        for row in rows:
            answer=row['expected'] if row['kind']!='EndState' else dict(answers=list(row['expected'].values()))
            row['raw_answer']=json.dumps(answer)
        score=event_score(rows)
        self.assertEqual(score['status'],'complete')
        for metric in ('ISR','LRA','EndState'):self.assertEqual(score[metric],1)
        for constant in ('yes','no'):
            for row in rows:
                answer={key:constant for key in row['expected']} if row['kind']!='EndState' else dict(answers=[constant]*len(row['facts']))
                row['raw_answer']=json.dumps(answer)
            score=event_score(rows)
            for metric in ('ISR','LRA','EndState'):self.assertEqual(score[metric],.5)

    def test_invalid_negative_answers_cannot_become_correct_by_inversion(self):
        for kind,answer,expected in [('ISR',dict(action='unknown',outcome='no',participants='yes'),dict(action='no',outcome='no',participants='yes')),
                                     ('LRA',dict(action='unknown'),dict(action='no')),
                                     ('EndState',dict(answers=['unknown']),{'1':'no'})]:
            with self.subTest(kind=kind):
                row=dict(kind=kind,raw_answer=json.dumps(answer),expected=expected,facts=['A fact.'])
                score=event_score([row])
                self.assertEqual(score['status'],'parse_error')
                self.assertEqual(score[kind],0)
                self.assertEqual(score['n_invalid'],1)
        score=event_score([dict(kind='EndState',facts=['A.','B.'],expected={'1':'no','2':'no'},raw_answer='{"answers":[]}')])
        self.assertEqual(score['status'],'parse_error');self.assertEqual(score['n_invalid'],2)

    def test_legacy_rows_without_expected_metadata_still_require_yes(self):
        rows=[dict(kind='ISR',raw_answer='{"action":"yes","outcome":"yes","participants":"yes"}'),
              dict(kind='LRA',raw_answer='{"action":"yes"}'),
              dict(kind='EndState',facts=['A fact.'],raw_answer='{"answers":["yes"]}')]
        score=event_score(rows)
        for metric in ('ISR','LRA','EndState'):self.assertEqual(score[metric],1)

    def test_preparation_reuses_the_same_evidence_for_each_factual_pair(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from worlddraft_eval.events.run import prepare_job
        from worlddraft_eval.gt.events import write_json
        spec=dict(events=[dict(id='land',action='The ball lands.',outcome='The ball is on the table.',participants=['ball'])],timeline=[],groups={'ball':'ball'},end_state=['The ball is on the table.'])
        with TemporaryDirectory() as directory:
            out=Path(directory);video=out/'video.mp4';video.write_bytes(b'video')
            write_json(out/'legacy/inputs/X.gt.json',{})
            contrasts=out/'contrasts.json'
            write_json(contrasts,dict(cases={'X':dict(ISR={'land':dict(action='The ball does not land.',outcome='The ball is not on the table.')},LRA={},EndState=['The ball is below the table.'])}))
            job=dict(case='X',path='job',video=str(video),size_bytes=video.stat().st_size,mtime_ns=video.stat().st_mtime_ns,
                     model='draft',cell='P2_AA')
            frames=[dict(index=72,time_s=4.5,path='same.jpg',sha256='same')]
            with patch('worlddraft_eval.events.run.NEGATIVE_QUESTIONS',contrasts), \
                 patch('worlddraft_eval.events.run.sample_video',return_value=([],{})), \
                 patch('worlddraft_eval.events.run.image_evidence',return_value=frames) as evidence:
                rows=prepare_job(job,out/'run',dict(cases={'X':spec},settings={}),dict(instances=[]),out/'legacy')
            self.assertEqual({r['kind'] for r in rows},{'ISR'})
            self.assertEqual(evidence.call_count,1)
            self.assertIs(rows[0]['frames'],rows[1]['frames'])
            self.assertNotEqual(rows[0]['prompt'],rows[1]['prompt'])

    def test_target_and_boundary_description_reaches_both_timing_stages(self):
        from worlddraft_eval.events.run import timing_prompt
        from worlddraft_eval.gt.events import PROTOCOL
        rule=load(PROTOCOL/'timing_rules.json')['rules']['A01'][0]
        frames=[dict(index=0,time_s=0),dict(index=12,time_s=.5)]
        row=dict(description=rule['description'],occurrence=1,required_boundaries=['onset','offset'],
                 frames=frames,fine={'onset':dict(frames=frames)},gt={'onset':dict(time_s=1.40625)})
        for boundary in (None,'onset'):
            prompt=timing_prompt(row,boundary)
            self.assertIn('bottom of the opening frame',prompt)
            self.assertIn('Onset:',prompt)
            self.assertIn('Offset:',prompt)
            self.assertIn('colors may differ',prompt)
            self.assertNotIn('1.40625',prompt)

    def test_all_timing_rules_explain_their_requested_boundary_types(self):
        from worlddraft_eval.gt.events import PROTOCOL
        labels={'interval':('Onset:','Offset:'),'hidden':('Onset:','Offset:'),
                'crossing':('Time:',),'onset':('Onset:',),'offset':('Offset:',)}
        for case,rules in load(PROTOCOL/'timing_rules.json')['rules'].items():
            for rule in rules:
                with self.subTest(case=case,event=rule['name']):
                    self.assertTrue(rule['description'].startswith('Target:'))
                    for label in labels[rule['take']]:self.assertIn(label,rule['description'])

    def test_end_state_uses_participants_without_revealing_source_motion(self):
        from worlddraft_eval.events.run import binary_prompt
        spec=dict(events=[],timeline=[],groups={'cups':'cup'},end_state=['The ball is visible.'],
                  target_context='Follow the cup initially on the left through the shuffle.')
        row=motion_requests(spec,{})[0];row['frames']=[]
        self.assertIn('Participants: cups',binary_prompt(row))
        self.assertNotIn(spec['target_context'],binary_prompt(row))
        self.assertEqual(row['target_context'],spec['target_context'])

    def test_old_run_cannot_be_relabeled_with_descriptions(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from worlddraft_eval.events.run import prepare, judge
        from worlddraft_eval.gt.events import write_json
        with TemporaryDirectory() as directory:
            out=Path(directory)
            write_json(out/'events/plan.json',dict(protocol='event-binary-earlier',timing_protocol='evaluated-motion-boundaries'))
            with self.assertRaisesRegex(ValueError,'archive and rebuild'):prepare(out)
            with self.assertRaisesRegex(ValueError,'different event protocol'):judge(out,0)

    def test_published_mapping_uses_duplicate_frames_and_pts(self):
        self.assertEqual(timestamp_interval(7,[1,1,4,4,7,7],[0,.0625,.125,.1875,.25,.3125]),[.1875,.25])
        self.assertIsNone(timestamp_interval(1,[1,4],[0,.0625]))
        self.assertIsNone(timestamp_interval(9,[1,4],[0,.0625]))

    def test_direction_and_hidden_boundaries(self):
        rule=dict(take='interval',direction=1,occurrence=1)
        result=boundaries([0,0,0,1,2,3,4,4,4],rule)
        self.assertEqual(result['boundaries'],{'onset':3,'offset':6})
        result=boundaries([1,1,0,0,0,1,1],dict(take='hidden',threshold=.1,occurrence=1))
        self.assertEqual(result['boundaries'],{'onset':2,'offset':5})

    def test_missing_event_cannot_receive_zero_error(self):
        row=dict(raw_answer='{"occurred":"no","time_index":null}',frames=[dict(index=0,time_s=0),dict(index=8,time_s=.5)],required_boundaries=['time'])
        score=timing_score(row)
        self.assertEqual(score['status'],'missing_event')
        self.assertIsNone(score['error_s'])
        self.assertEqual(parse_timing('{"occurred":"yes","index":7}',row['frames'],['index'])['status'],'parse_error')

    def test_error_uses_evidence_interval_midpoints(self):
        frames=[dict(index=0,time_s=0),dict(index=8,time_s=.5)]
        row=dict(raw_answer='{"occurred":"yes","time_index":8}',frames=frames,required_boundaries=['time'],
                 fine={'time':dict(frames=[dict(index=4,time_s=.25),dict(index=6,time_s=.375)],raw_answer='{"occurred":"yes","index":6}')},
                 gt={'time':dict(time_s=.2,interval_s=[.1875,.2125])})
        self.assertAlmostEqual(timing_score(row)['error_s'],.1125)
        row['fine']['time']['raw_answer']='{"occurred":"yes","index":4}'
        self.assertEqual(timing_score(row)['status'],'truncated_prediction')

    def test_reversed_boundaries_are_invalid(self):
        frames=[dict(index=i,time_s=i*.5) for i in range(4)]
        self.assertEqual(parse_timing('{"occurred":"yes","onset_index":3,"offset_index":1}',frames,['onset_index','offset_index'])['status'],'parse_error')
        row=dict(raw_answer='{"occurred":"yes","onset_index":1,"offset_index":3}',frames=frames,
                 required_boundaries=['onset','offset'],gt={b:dict(time_s=.5,interval_s=[.4,.6]) for b in ['onset','offset']},
                 fine={b:dict(frames=frames,raw_answer='{"occurred":"yes","index":'+str(i)+'}') for b,i in [('onset',3),('offset',1)]})
        result=timing_score(row)
        self.assertEqual(result['status'],'parse_error')
        self.assertIsNone(result['error_s'])

    @unittest.skipUnless(paths.has_data(), 'benchmark data not downloaded')

    def test_revised_a07_canvas_uses_preprocessing(self):
        from worlddraft_eval.gt.events import ROOT
        from worlddraft_eval.common.spatial_mapping import input_crop
        gt=load(paths.GT/'interaction-regions/A07.json')
        self.assertEqual(canonical_size(gt),[832,480])
        self.assertEqual(input_crop('UniVideo',canonical_size(gt),[832,480]),[0,0,832,480])

    @unittest.skipUnless(paths.has_data(), 'benchmark data not downloaded')

    def test_target_outside_model_crop_retains_full_frame(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        import numpy as np
        from PIL import Image
        from worlddraft_eval.events.run import image_evidence
        gt=dict(case='D01',width=832,height=480,frames=[dict(boxes={'target':dict(visible=True,box=[828,200,831,250])})])
        sample=(np.zeros((480,832,3),dtype=np.uint8),0,0.)
        with TemporaryDirectory() as directory, patch('worlddraft_eval.events.run.canonical_size',return_value=[832,480]), patch('worlddraft_eval.gt.visible.preprocessing_transform',return_value=dict(resized_size=[832,480],offset_xy=[0,0])):
            frames=image_evidence('unused.mp4',[0],Path(directory),'test',gt,['target'],'VINO',samples={0.:sample})
            with Image.open(frames[0]['path']) as picture:
                self.assertEqual(picture.size,(832,480))

    def test_unknown_is_parse_error_in_new_binary_protocol(self):
        row=dict(kind='LRA',raw_answer='{"action":"unknown"}')
        self.assertEqual(event_score([row])['status'],'parse_error')
        row['raw_answer']='{"action":"yes"}'
        score=event_score([row]);self.assertEqual(score['LRA'],1)
        self.assertNotIn('LRA_strict',score)




def test_frozen_requests_ignore_answers_but_detect_question_changes():
    from worlddraft_eval.events.run import request_fingerprint
    rows=[dict(kind='ISR',window=[0.0,1.0],frames=[],raw_answer=None)]
    original=request_fingerprint(rows)
    rows[0]['raw_answer']='changed response'
    assert request_fingerprint(rows)==original
    rows[0]['window'][0]+=.1
    assert request_fingerprint(rows)!=original


def test_row_answers_keep_unknown_and_mark_invalid():
    from worlddraft_eval.events.run import row_answers
    isr=dict(kind='ISR',raw_answer='{"action":"yes","outcome":"unknown","participants":"maybe"}')
    assert row_answers(isr)=={'action':'yes','outcome':'unknown','participants':'invalid'}
    end=dict(kind='EndState',facts=['a','b'],raw_answer='{"answers":["no"]}')
    assert row_answers(end)=={'1':'invalid','2':'invalid'}
    assert row_answers(dict(kind='ISR',raw_answer=None))=={}


if __name__=='__main__':unittest.main()
