"""ELLA-format questions, full frames + GT zooms, and auditable per-case RF/AP/SL/count results."""
from __future__ import annotations

from collections import defaultdict
import csv
import fcntl
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw

from worlddraft_eval.common import io as Q, paths
from worlddraft_eval.common.io import write_json

# Helpers frozen into each QA run next to the settings and judge code.
HELPERS = {'evidence': 'qa.evidence', 'spatial_mapping': 'common.spatial_mapping'}

FIELDS = 'item_id keywords proposition_id dependency category_broad category_detailed tuple question_natural_language attribute entity_id metric expected_answer applies_to'.split()
ATTRIBUTES = ('color', 'material', 'texture', 'environment', 'lighting')
ANSWER_BINARY = 'Answer yes or no.'
ANSWER_COUNT = 'Answer with a nonnegative integer.'
REPORT_VERSION = 'literal-ap-gated'


def read_json(path):
    return json.loads(Path(path).read_text())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_questions(path, config, gt):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if list(reader.fieldnames or []) != FIELDS:
            raise ValueError('Expected the documented ELLA + attribute CSV columns')
        rows = [r for r in reader if r['item_id'] == config['episode']]
    ids = set()
    groups = config['qa']['groups'][config['episode']]
    known = set(gt['entities']) | set(groups)
    for members in groups.values():
        if not members or not set(members) <= set(gt['entities']):
            raise ValueError('Unknown group member')
    for r in rows:
        pid, metric = r['proposition_id'], r['metric']
        if None in r or r['item_id'] != config['episode'] or not pid.isdigit() or int(pid) < 1 or pid in ids:
            raise ValueError('Invalid/duplicate question identifier')
        ids.add(pid)
        if r['dependency'] != '0':
            dep = next((x for x in rows if x['proposition_id'] == r['dependency']), None)
            if (metric != 'AP' or dep is None or dep['metric'] != 'AP' or dep['dependency'] != '0'
                    or dep['proposition_id']==pid or any(dep[k]!=r[k] for k in ('entity_id','attribute','expected_answer','applies_to','tuple'))):
                raise ValueError('Invalid AP presence dependency')
        if metric not in ('EntityCount', 'RF', 'AP', 'SL') or not r['question_natural_language'].strip():
            raise ValueError('Invalid metric or empty question')
        if r['entity_id'] and r['entity_id'] not in known:
            raise ValueError('Unknown entity/group')
        if metric == 'EntityCount':
            scope = config['qa']['bindings'][r['item_id']][pid]['count_scope']
            if scope not in ('global', 'region') or r['attribute'] or r['expected_answer']:
                raise ValueError('Invalid count question')
            if (scope == 'region' and r['entity_id'] not in gt['entities']) or not r['entity_id']:
                raise ValueError('Count target missing or not a single region entity')
        elif r['attribute'] not in ATTRIBUTES or r['category_detailed'] != r['attribute']:
            raise ValueError('Invalid attribute classification')
        elif r['expected_answer'] not in (('no',) if metric == 'SL' else ('yes', 'no')):
            raise ValueError('Invalid expected answer')
        if r['applies_to'] not in ('ALL', 'AC') or (metric == 'SL' and (r['attribute'] != 'color' or r['applies_to'] != 'AC')):
            raise ValueError('SL must be AC-only source color')
    if not rows:
        raise ValueError('Empty question set')
    return rows


def visible_gt(sparse, dense, entity_labels=None):
    """Reuse actual Blender instance masks at their exported instants; keep the field names of the entity GT."""
    from worlddraft_eval.gt.visible import letterbox_transform
    tr = sparse.get('letterbox') or letterbox_transform(sparse['source_size'], [dense['width'],dense['height']])
    sw, sh = sparse['source_size']; rw, rh = tr['resized_size']; ox, oy = tr['offset_xy']
    affine = np.array([[rw/sw,0,ox],[0,rh/sh,oy]])
    sw, sh = sparse['source_size']
    frames = []
    for f in sparse['frames']:
        boxes = {}
        for e in f['entities']:
            visible = e['visible'] and e['mask_area'] > 0
            b = e['bbox_xyxy']
            mapped = (np.array([[b[0]*sw, b[1]*sh, 1], [b[2]*sw, b[3]*sh, 1]]) @ affine.T).ravel().tolist() if visible else None
            center = (np.array([e['center_xy'][0]*sw, e['center_xy'][1]*sh, 1]) @ affine.T).tolist() if visible else None
            boxes[e['entity_id']] = dict(box=mapped, visible=visible, visible_mask_area=e['mask_area'], center=center)
        frames.append(dict(frame=f['model_input_index'], time=f['model_input_time_s'], source_frame=f['source_index'], blender_frame=f['blender_frame'], boxes=boxes))
    if entity_labels is not None:
        if set(boxes) != set(entity_labels):
            raise ValueError('QA labels must describe every visible-mask entity')
        entities = {eid: dict(dense['entities'].get(eid, {}), label=label) for eid,label in entity_labels.items()}
    else:
        entities = dense['entities']
    if set(boxes) != set(entities):
        raise ValueError('Sparse and dense GT entity sets differ')
    return dict(schema='worlddraft.gt', episode=dense['episode'], camera=dense['camera'], fps=dense['fps'],
                frame_count=dense['frame_count'], width=dense['width'], height=dense['height'], entities=entities, frames=frames,
                source=dict(sparse_source=sparse['source_blend'], source_cell=sparse['source_cell']),
                adaptation=dict(sampling='sparse exported instants only', box_type='visible_instance_mask_bbox',
                                visible='at least one rendered instance-mask pixel', canvas_mapping=affine.tolist()))


def count_gt(frame, targets, scope, eid):
    visible = [x for x in targets if frame['boxes'][x]['visible']]
    if scope == 'global':
        return len(visible)
    b = frame['boxes'][eid]['box']
    return sum(b[0] <= frame['boxes'][x]['center'][0] < b[2]
               and b[1] <= frame['boxes'][x]['center'][1] < b[3] for x in visible)


def save_image(path, im, run):
    im.save(path, quality=95)
    return dict(path=str(path.relative_to(run)), sha256=Q.sha256(path), size=list(im.size))


def rf_time(times, by_time, targets):
    # Select from reference GT only, never from generated appearance or answers.
    return max(sorted(times), key=lambda t: sum(
        by_time[t]['boxes'][eid]['visible_mask_area']
        for eid in targets if by_time[t]['boxes'][eid]['visible']))


def make_evidence(frame, gf, gt, directory, run, crop_side, entity_ids=None):
    im = Image.fromarray(frame)
    w, h = im.size
    scale = [w/gt['width'], h/gt['height']]
    full = save_image(directory / f"f{gf['frame']:03d}_full.jpg", im, run)
    regions = {}
    for eid, entry in gf['boxes'].items():
        if not entry['visible'] or (entity_ids is not None and eid not in entity_ids):
            continue
        b = [v*scale[i % 2] for i, v in enumerate(entry['box'])]
        crop_box = [max(0, math.floor(b[0])), max(0, math.floor(b[1])), min(w, math.ceil(b[2])), min(h, math.ceil(b[3]))]
        if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
            raise ValueError('Empty GT crop')
        ctx = im.copy()
        ImageDraw.Draw(ctx).rectangle(b, outline='white', width=1)
        ImageDraw.Draw(ctx).text((b[0], max(0, b[1]-12)), eid, fill='white', stroke_fill='black', stroke_width=1)
        crop = im.crop(crop_box)
        factor = crop_side/max(crop.size)
        crop = crop.resize((max(1, round(crop.width*factor)), max(1, round(crop.height*factor))), Image.Resampling.LANCZOS)
        stem = directory / f"f{gf['frame']:03d}_{eid}"
        regions[eid] = dict(gt_box=entry['box'], video_box=b, crop_box=crop_box,
            context=save_image(stem.with_name(stem.name+'_context.jpg'), ctx, run),
            zoom=save_image(stem.with_name(stem.name+'_zoom.jpg'), crop, run))
    return dict(full=full, regions=regions, video_size=[w,h], gt_size=[gt['width'],gt['height']], mapping='whole_condition_canvas_resize')


def prepare(settings_path, input_dir, model, cells, out, run_id, video_paths=None):
    config = read_json(settings_path)
    sources = {k: paths.settings_file(settings_path, v) for k, v in config['files'].items()}
    sparse = read_json(sources['visibility_gt'])
    gt = visible_gt(sparse, read_json(sources['entities_gt']), config['qa'].get('entity_labels'))
    questions = load_questions(sources['questions'], config, gt)
    if any(not re.fullmatch(r'P[012]_(AN|AA|AC)', c) for c in cells) or len(set(cells)) != len(cells):
        raise ValueError('Invalid/duplicate cells')
    if run_id is not None and (not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id) or run_id in ('.','..')):
        raise ValueError('Invalid run identifier')
    videos = {c: Path(video_paths[c] if video_paths is not None else input_dir/c/'output.mp4').resolve() for c in cells}
    for path in videos.values():
        if not path.is_file():
            raise ValueError(f'Missing video: {path}')
    run = out/'runs'/run_id if run_id is not None else out
    (run/'inputs').mkdir(parents=True, exist_ok=False)
    frozen = {}
    for name, src in dict(sources, settings=settings_path, code=Path(__file__), io=Path(Q.__file__), engine=Q.module_file('common.judge')).items():
        dst = run/'inputs'/f'{name}{src.suffix}'
        shutil.copyfile(src, dst)
        frozen[str(dst.relative_to(run))] = Q.sha256(dst)
    write_json(run/'inputs/gt.json', gt)
    frozen['inputs/gt.json'] = Q.sha256(run/'inputs/gt.json')
    if config['qa'].get('visibility_from_masks'):
        for name,module in HELPERS.items():
            dst=run/'inputs'/f'{name}.py'
            shutil.copyfile(Q.module_file(module),dst)
            frozen[str(dst.relative_to(run))]=Q.sha256(dst)
        for f in sparse['frames']:
            dst = run/'inputs'/f"mask_{f['model_input_index']:03d}.png"
            shutil.copyfile(Q.repo_path(f['instance_mask_path']),dst)
            frozen[str(dst.relative_to(run))] = Q.sha256(dst)
    times = config['qa']['times_s']
    by_time = {f['time']: f for f in gt['frames']}
    if not set(times) <= set(by_time) or not set(config['qa']['count_times_s']) <= set(times):
        raise ValueError('Sampling must use actual exported visible GT instants')
    groups = config['qa']['groups'][config['episode']]
    all_ids = list(gt['entities'])
    evidence_ids = None
    if config['qa'].get('global_count_view')=='full_frame':
        evidence_ids = set()
        for q in questions:
            binding=config['qa'].get('bindings',{}).get(q['item_id'],{}).get(q['proposition_id'],{})
            if q['metric']=='EntityCount' and binding.get('count_scope')=='global':continue
            evidence_ids.update(groups.get(q['entity_id'],[q['entity_id']] if q['entity_id'] else []))
    jobs = []
    for cell, video in videos.items():
        directory = run/cell
        evdir = directory/'evidence'
        evdir.mkdir(parents=True)
        decoded, fps = Q.sample_video(video, times)
        video_gt = gt
        if config['qa'].get('visibility_from_masks'):
            from worlddraft_eval.qa.evidence import output_mask_gt
            video_gt = output_mask_gt(sparse,gt,model,decoded[0][0].shape[1::-1],run/'inputs')
        video_by_time = {f['time']:f for f in video_gt['frames']}
        write_json(directory/'gt.json',video_gt)
        records = {}
        if len({i for _,i,_ in decoded}) != len(times):
            raise ValueError('Duplicate sampled video frames')
        for t,(frame,fi,actual) in zip(times,decoded):
            records[t] = dict(requested_time=t, video_frame=fi, video_time=actual, gt_frame=by_time[t]['frame'], gt_time=t,
                              source_frame=by_time[t]['source_frame'], blender_frame=by_time[t]['blender_frame'],
                              **make_evidence(frame,video_by_time[t],video_gt,evdir,run,config['qa']['crop_side'],evidence_ids))
        write_json(directory/'evidence.json', records)
        requests=[]
        for q in questions:
            if q['applies_to'] not in ('ALL',cell.split('_')[1]):
                continue
            binding=config['qa'].get('bindings',{}).get(q['item_id'],{}).get(q['proposition_id'],{})
            if binding.get('cells') and cell not in binding['cells']:continue
            if binding.get('question_by_cell'):
                q=dict(q,question_natural_language=binding['question_by_cell'].get(cell,q['question_natural_language']))
            metric,eid=q['metric'],q['entity_id']
            targets=groups.get(eid,[eid] if eid else [])
            scope=config['qa']['bindings'][q['item_id']][q['proposition_id']]['count_scope'] if metric=='EntityCount' else ''
            instants=config['qa']['count_times_s'] if metric=='EntityCount' else times
            persistence = metric=='AP' and (q['dependency']!='0' or not any(x['metric']=='AP' and x['dependency']!='0' for x in questions))
            if metric=='RF' or (metric=='AP' and not persistence):instants=[rf_time(times,by_time,targets)]
            sequences=[instants] if persistence else [[t] for t in instants]
            for ts in sequences:
                frames=[]; images=[]
                for t in ts:
                    rec=records[t]
                    present=[x for x in targets if x in rec['regions']]
                    if targets and scope!='global' and not present:
                        continue
                    selected=[]
                    if scope=='global':
                        selected=[('Full image',rec['full'])]
                        if config['qa'].get('global_count_view')!='full_frame':
                            selected += [(f'Zoom of {x} in the full image',rec['regions'][x]['zoom']) for x in present]
                    elif scope=='region':
                        selected=[('Full image',rec['regions'][eid]['context']),('Zoom of the marked region in the full image',rec['regions'][eid]['zoom'])]
                    elif eid:
                        selected=[('Full image',rec['regions'][present[0]]['context'])]+[(f'Zoom of {x} in the full image',rec['regions'][x]['zoom']) for x in present]
                    else:
                        selected=[('Full image',rec['full'])]
                    frames.append(dict(requested_time=t,video_frame=rec['video_frame'],video_time=rec['video_time'],gt_frame=rec['gt_frame'],gt_time=t,
                                       regions={x:{k:rec['regions'][x][k] for k in ('gt_box','video_box','crop_box')} for x in present}))
                    images.extend(dict(label=f'{label}; time {rec["video_time"]:.4f}s',**im) for label,im in selected)
                sample='sequence' if persistence else f't{ts[0]:.4f}'
                eligible=len(frames)>= (2 if persistence else 1)
                expected=q['expected_answer']
                if metric=='EntityCount' and eligible:
                    category_ids=[x for x in all_ids if gt['entities'][x]['label']==gt['entities'][targets[0]]['label']]
                    expected=count_gt(video_by_time[ts[0]],category_ids,scope,eid)
                prompt=q['question_natural_language']+'\n'+(ANSWER_COUNT if metric=='EntityCount' else ANSWER_BINARY)
                requests.append(dict(**q,model=model,cell=cell,sample_id=sample,count_scope=scope,expected=expected,
                    frames=frames,images=images,prompt=prompt,eligible=eligible,raw_answer=None))
        write_json(directory/'answers.json',requests)
        jobs.append(dict(cell=cell,path=str(directory.relative_to(run)),video=str(video),video_sha256=Q.sha256(video),fps=fps,
                         request_sha256=fingerprint([{k:v for k,v in r.items() if k!='raw_answer'} for r in requests]),requests=len(requests)))
        print(f'Prepared {cell}: {len(requests)} requests',flush=True)
    plan=dict(protocol=config['protocol'],episode=config['episode'],model=model,run_id=run_id,inputs=frozen,
              code_sha256=Q.sha256(__file__),jobs=jobs,score_rules=config['qa'],status='prepared')
    write_json(run/'plan.json',plan)
    report(out,run_id)
    return run


def parse(row):
    if not row['eligible']:return None,'gt_not_visible'
    raw=row['raw_answer']
    if raw is None:return None,'pending'
    value=raw.strip().lower().rstrip('.!。！').strip()
    if row['metric']=='EntityCount':
        return (int(value),'answered') if re.fullmatch(r'\d+',value) else (None,'parse_error')
    return (value,'answered') if value in ('yes','no') else (None,'parse_error')


def row_score(row):
    value,status=parse(row)
    if status != 'answered':return None
    return int(value==('yes' if row['metric']=='SL' else row['expected']))


def dependency_row(row, rows):
    matches=[candidate for candidate in rows if candidate['proposition_id']==row['dependency']]
    if len(matches)!=1:raise ValueError('Missing or ambiguous AP dependency')
    parent=matches[0]
    if (row['metric']!='AP' or parent['metric']!='AP' or parent.get('dependency','0')!='0'
            or any(parent[k]!=row[k] for k in ('entity_id','attribute','expected'))
            or any(parent.get(k)!=row.get(k) for k in ('tuple','applies_to','cell','model'))):
        raise ValueError('Invalid AP dependency binding')
    return parent


def score_state(row, rows):
    if row.get('dependency','0')!='0':
        parent=dependency_row(row,rows)
        parent_score=row_score(parent)
        if parent_score is None:return None,parse(parent)[1]
        if parent_score==0:return 0,'answered'
    return row_score(row),parse(row)[1]


def gated_row_score(row, rows):
    return score_state(row,rows)[0]


def judge(out,run_id,worker=0,workers=1,batch_size=8,engine=None):
    run=out/'runs'/run_id if run_id is not None else out
    plan=read_json(run/'plan.json')
    if workers<1 or not 0<=worker<workers or batch_size<1:raise ValueError('Invalid worker/batch')
    if Q.sha256(__file__)!=plan['code_sha256']:raise ValueError('Runner changed after prepare; create a new run')
    for name,sha in plan['inputs'].items():
        if Q.sha256(run/name)!=sha:raise ValueError(f'Frozen input changed: {name}')
    for key,module in [('io',Q.__file__),('engine',Q.module_file('common.judge'))]:
        if Q.sha256(module)!=plan['inputs'][f'inputs/{key}.py']:raise ValueError(f'Helper changed: {key}')
    for key,module in HELPERS.items():
        if f'inputs/{key}.py' in plan['inputs'] and Q.sha256(Q.module_file(module))!=plan['inputs'][f'inputs/{key}.py']:
            raise ValueError(f'Helper changed: {key}')
    for i,job in enumerate(plan['jobs']):
        if i%workers!=worker:continue
        directory=run/job['path']
        with (directory/'judge.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            rows=read_json(directory/'answers.json')
            if fingerprint([{k:v for k,v in r.items() if k!='raw_answer'} for r in rows])!=job['request_sha256']:
                raise ValueError('Prepared requests changed')
            pending=[r for r in rows if r['eligible'] and r['raw_answer'] is None]
            for im in {im['path']:im for r in pending for im in r['images']}.values():
                if Q.sha256(run/im['path'])!=im['sha256']:raise ValueError('Evidence changed')
            if pending and engine is None:
                from worlddraft_eval.common.judge import Engine
                engine=Engine()
            if pending:write_json(directory/'judge.json',dict(engine.metadata,max_tokens=16,enable_thinking=False,output_format='constrained yes/no or nonnegative integer'))
            for start in range(0,len(pending),batch_size):
                batch=pending[start:start+batch_size];contents=[]
                for r in batch:
                    content=[]
                    for im in r['images']:
                        with Image.open(run/im['path']) as image:
                            content.extend([dict(type='text',text=im['label']),dict(type='image',image=image.convert('RGB'))])
                    content.append(dict(type='text',text=r['prompt']));contents.append(content)
                if hasattr(engine,'params'):
                    from vllm import SamplingParams
                    from vllm.sampling_params import StructuredOutputsParams
                    engine.params=[SamplingParams(temperature=0,max_tokens=16,structured_outputs=StructuredOutputsParams(
                        regex=r'(0|[1-9][0-9]*)' if r['metric']=='EntityCount' else r'(yes|no)')) for r in batch]
                responses=engine.run(contents)
                if len(responses)!=len(batch):raise ValueError('Judge response count mismatch')
                for r,raw in zip(batch,responses):r['raw_answer']=raw
                write_json(directory/'answers.json',rows)
                print(f'{job["cell"]}: {min(start+batch_size,len(pending))}/{len(pending)}',flush=True)


def aggregate(rows,metric,attr=None,count_scope=None):
    selected=[r for r in rows if r['metric']==metric and (attr is None or r['attribute']==attr)
              and (count_scope is None or r['count_scope']==count_scope)]
    if metric=='AP' and any(r.get('dependency','0')!='0' for r in selected):selected=[r for r in selected if r.get('dependency','0')!='0']
    eligible=[r for r in selected if r['eligible']]
    states=[score_state(r,rows)[1] for r in eligible]
    status='not_applicable' if not eligible else ('pending' if 'pending' in states else ('parse_error' if 'parse_error' in states else 'gt_not_visible' if 'gt_not_visible' in states else 'complete'))
    by_q=defaultdict(list)
    for r in eligible:
        score = gated_row_score(r, rows)
        if score is not None:by_q[r['proposition_id']].append(score)
    if by_q and status == 'complete':
        if metric == 'EntityCount' and count_scope == 'global':
            # A global count proposition is a video-level invariant: every
            # sampled frame must be exact before the proposition earns 1.
            value=sum(int(all(score == 1 for score in scores)) for scores in by_q.values()) / len(by_q)
        else:
            value=sum(sum(v)/len(v) for v in by_q.values())/len(by_q)
    else:
        value=None
    return dict(value=value,status=status,n_expected=len(eligible),n_scored=sum(map(len,by_q.values())),n_error=states.count('parse_error'),n_questions=len(by_q))


def report(out,run_id):
    run=out/'runs'/run_id if run_id is not None else out
    plan=read_json(run/'plan.json');videos=[];details=[];pages=[]
    evidence_prefix=f'runs/{run_id}/' if run_id is not None else ''
    for job in plan['jobs']:
        rows=read_json(run/job['path']/'answers.json')
        values=dict(model=plan['model'],item_id=plan['episode'],cell=job['cell'],run_id=run_id)
        values.update(DF_total=None,DF_total_status='not_computed',DF_total_coverage=None)
        for scope,label in [('global','global'),('region','local')]:
            key='DF_count_'+label+'_accuracy';result=aggregate(rows,'EntityCount',count_scope=scope)
            values[key]=result['value']
            values[key+'_status']=result['status']
            values[key+'_coverage']=result['n_scored']/result['n_expected'] if result['n_expected'] else None
        for metric in ('RF','AP','SL'):
            for attr in [None,*ATTRIBUTES] if metric!='SL' else ['color']:
                key=metric+'_'+(attr or 'total');result=aggregate(rows,metric,attr)
                values[key]=result['value']
                values[key+'_status']=result['status']
                values[key+'_coverage']=result['n_scored']/result['n_expected'] if result['n_expected'] else None
        values['n_pending']=sum(parse(r)[1]=='pending' for r in rows)
        values['n_error']=sum(parse(r)[1]=='parse_error' for r in rows)
        videos.append(values)
        review=[]
        for r in rows:
            answer,status=parse(r)
            row=dict(model=r['model'],item_id=r['item_id'],cell=r['cell'],proposition_id=r['proposition_id'],sample_id=r['sample_id'],metric=r['metric'],attribute=r['attribute'],entity_id=r['entity_id'],count_scope=r['count_scope'],question=r['question_natural_language'],expected_answer=r['expected'],answer=answer,score=gated_row_score(r,rows),status=score_state(r,rows)[1],
                count_error=answer-r['expected'] if r['metric']=='EntityCount' and status=='answered' else None,
                evidence_record=f"{evidence_prefix}{job['path']}/answers.json",images=[im['path'] for im in r['images']])
            details.append(row)
            images=''.join(f'<figure><img loading="lazy" src="{html.escape(im["path"])}"><figcaption>{html.escape(im["label"])}</figcaption></figure>' for im in r['images'])
            review.append(f'<details><summary>{html.escape(r["proposition_id"]+" / "+r["sample_id"]+" / "+r["metric"]+" / "+r["question_natural_language"])} — {html.escape(str(answer))} ({status})</summary><p>GT/expected: {html.escape(str(r["expected"]))}; score: {gated_row_score(r,rows)}</p>{images}</details>')
        pages.append('<h2>'+html.escape(job['cell'])+'</h2>'+''.join(review))
    Q.write_csv(out/'per_video.csv',videos)
    Q.write_csv(out/'answers.csv',details)
    summary=[]
    for key in [k for k in videos[0] if k.endswith('_total') or k in ('DF_count_global_accuracy','DF_count_local_accuracy')
                or any(k==m+'_'+a for m in ('RF','AP','SL') for a in ATTRIBUTES)]:
        active=[v for v in videos if v[key+'_status']!='not_applicable'];valid=[v[key] for v in active if v[key] is not None]
        status='not_applicable' if not active else ('complete' if len(valid)==len(active) else 'incomplete')
        if active and all(v[key+'_status']=='not_computed' for v in active):status='not_computed'
        summary.append(dict(model=plan['model'],item_id=plan['episode'],metric=key,value=sum(valid)/len(valid) if valid and len(valid)==len(active) else None,unit='ratio',n_videos=len(valid),n_expected_videos=len(active),status=status))
    Q.write_csv(out/'summary.csv',summary)
    write_json(out/'provenance.json',dict(protocol=plan['protocol'],source_run=f'runs/{run_id}' if run_id is not None else '.',inputs=plan['inputs'],videos=plan['jobs'],
        report_version=REPORT_VERSION,report_code_sha256=Q.sha256(__file__),answer_reuse=plan.get('answer_reuse'),
        score_rules=dict(plan['score_rules'],count_scoring='exact match; global proposition scores 1 only when every sampled frame is exact; local remains per-frame; mean per question then per video',
                         DF_total='not computed; count accuracy is only an Entity submetric')))
    (run/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>'+html.escape(plan['episode'])+' QA evidence</title><style>body{font:16px sans-serif;max-width:1300px;margin:30px auto}details{border-bottom:1px solid #ddd;padding:8px}summary{cursor:pointer}figure{display:inline-block;vertical-align:top;margin:8px}img{max-width:720px;max-height:300px}figcaption{font-size:12px}</style><h1>'+html.escape(plan['episode'])+' QA: questions, answers and evidence</h1>'+''.join(pages))
    return videos
