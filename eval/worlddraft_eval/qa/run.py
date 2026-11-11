"""Prepare, resume and summarize the complete frozen CSV QA benchmark."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
from pathlib import Path
import time
import traceback

from worlddraft_eval.qa import engine as D
from worlddraft_eval.common.videos import load_inventory

DEFAULT_PROTOCOL = Path(__file__).resolve().parents[2]/'protocol/questions'
RUN_NAME = 'qa'


def run_dir(out):
    return Path(out)/RUN_NAME


def prepare_group(job):
    directory = Path(job['directory'])
    if (directory/'plan.json').exists():
        plan=D.read_json(directory/'plan.json')
        assert plan['model']==job['model'] and plan['episode']==job['case']
        for name,sha in plan['inputs'].items():
            assert D.Q.sha256(directory/name)==sha, name
        assert D.Q.sha256(job['settings'])==plan['inputs']['inputs/settings.json']
        return
    if directory.exists():
        # Preserve evidence from interrupted preparation before rebuilding this group.
        directory.rename(directory.with_name(directory.name+f'.incomplete-{time.time_ns()}'))
    for row in job['videos']:
        stat=Path(row['video']).stat()
        if stat.st_size!=row['size_bytes'] or stat.st_mtime_ns!=row['mtime_ns']:
            raise ValueError(f"Video changed since inventory: {row['video']}")
    D.prepare(Path(job['settings']),None,job['model'],[v['cell'] for v in job['videos']],
              directory,None,video_paths={v['cell']:v['video'] for v in job['videos']})


def prepare(out,inventory,protocol_dir=DEFAULT_PROTOCOL,workers=4):
    root=run_dir(out);root.mkdir(parents=True,exist_ok=True)
    plan_path=root/'plan.json'
    if plan_path.exists():
        plan=D.read_json(plan_path)
        if plan['inventory_sha256']!=D.Q.sha256(inventory):
            raise ValueError('Inventory changed; use a new result directory')
    else:
        groups=defaultdict(list)
        for row in load_inventory(inventory):
            if row['status']=='available':groups[row['model'],row['case']].append(row)
        jobs=[]
        for (model,case),videos in sorted(groups.items()):
            settings=(Path(protocol_dir)/f'{case}.json').resolve()
            if not settings.is_file():raise ValueError(f'Missing QA settings: {settings}')
            jobs.append(dict(model=model,case=case,videos=videos,settings=str(settings),
                             directory=str((root/'videos'/model/case).resolve())))
        plan=dict(protocol='full-entity',created_at=time.time(),inventory=str(Path(inventory).resolve()),
                  inventory_sha256=D.Q.sha256(inventory),jobs=jobs)
        D.write_json(plan_path,plan)
    errors=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(prepare_group,j):j for j in plan['jobs']}
        for future in as_completed(futures):
            job=futures[future]
            try:future.result()
            except Exception as exc:
                errors.append(dict(model=job['model'],case=job['case'],error=repr(exc),traceback=traceback.format_exc()))
                print(f"PREPARE ERROR {job['model']}/{job['case']}: {exc}",flush=True)
    D.write_json(root/'prepare_status.json',dict(updated_at=time.time(),groups=len(plan['jobs']),errors=errors))
    report(out)
    if errors:raise RuntimeError(f'{len(errors)} QA groups failed; see prepare_status.json')


def pending(directory):
    plan=D.read_json(directory/'plan.json')
    return sum(D.parse(r)[1]=='pending' for job in plan['jobs']
               for r in D.read_json(directory/job['path']/'answers.json'))


def judge(out,worker=0,workers=4,batch_size=16,model_path=None,engine=None,job_list=None):
    if workers<1 or not 0<=worker<workers or batch_size<1:raise ValueError('Invalid worker/batch size')
    root=run_dir(out);plan=D.read_json(root/'plan.json')
    assigned = set(D.read_json(job_list)[worker]) if job_list else None
    with (root/f'worker-{worker}.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        errors=[];completed=0
        for i,job in enumerate(plan['jobs']):
            if (i not in assigned) if assigned is not None else (i%workers!=worker):continue
            directory=Path(job['directory'])
            try:
                if pending(directory) and engine is None:
                    from worlddraft_eval.common.judge import Engine, MODEL_PATH
                    engine=Engine(model_path or MODEL_PATH)
                D.judge(directory,None,batch_size=batch_size,engine=engine)
                D.report(directory,None)
                completed+=1
            except Exception as exc:
                errors.append(dict(model=job['model'],case=job['case'],error=repr(exc),traceback=traceback.format_exc()))
                print(f"JUDGE ERROR {job['model']}/{job['case']}: {exc}",flush=True)
            D.write_json(root/f'worker-{worker}.json',dict(worker=worker,workers=workers,
                completed_groups=completed,last_group=f"{job['model']}/{job['case']}",updated_at=time.time(),errors=errors))
        if errors:raise RuntimeError(f'{len(errors)} QA groups failed; see worker-{worker}.json')


def snapshot_rows(directory):
    """Read atomic answer checkpoints without overwriting active group reports."""
    plan=D.read_json(directory/'plan.json');videos=[]
    metrics=[('DF_count_global_accuracy','EntityCount',None,'global'),
             ('DF_count_local_accuracy','EntityCount',None,'region')]
    metrics += [(metric+'_'+(attr or 'total'),metric,attr,None)
                for metric in ('RF','AP','SL')
                for attr in ([None,*D.ATTRIBUTES] if metric!='SL' else ['color'])]
    for job in plan['jobs']:
        rows=D.read_json(directory/job['path']/'answers.json')
        values=dict(model=plan['model'],item_id=plan['episode'],cell=job['cell'],run_id=None,
                    DF_total=None,DF_total_status='not_computed',DF_total_coverage=None)
        for key,metric,attr,scope in metrics:
            result=D.aggregate(rows,metric,attr,count_scope=scope)
            values.update({key:result['value'],key+'_status':result['status'],
                           key+'_coverage':result['n_scored']/result['n_expected'] if result['n_expected'] else None})
        values['n_pending']=sum(D.parse(r)[1]=='pending' for r in rows)
        values['n_error']=sum(D.parse(r)[1]=='parse_error' for r in rows)
        videos.append(values)
    return videos


def report(out,snapshot=False):
    root=run_dir(out);plan=D.read_json(root/'plan.json');rows=[];counts=defaultdict(int)
    sources=[]
    for job in plan['jobs']:
        directory=Path(job['directory'])
        if not (directory/'plan.json').exists():
            counts['unprepared_videos']+=len(job['videos']);continue
        values=snapshot_rows(directory) if snapshot else D.report(directory,None)
        for row in values:
            row['case']=row.pop('item_id')
            row['protocol']=plan['protocol']
            row['source']=str(directory/row['cell']/'answers.json')
            row['run_id']=RUN_NAME
            rows.append(row)
            counts['complete_videos' if not row['n_pending'] and not row['n_error'] else 'incomplete_videos']+=1
            counts['pending_answers']+=row['n_pending'];counts['parse_errors']+=row['n_error']
        sources.append(dict(path=str(directory/'plan.json'),sha256=D.Q.sha256(directory/'plan.json')))
    if rows:D.Q.write_csv(Path(out)/'qa_per_video.csv',rows)
    D.write_json(root/'status.json',dict(updated_at=time.time(),**counts))
    D.write_json(Path(out)/'qa_provenance.json',dict(protocol=plan['protocol'],source_run=str(root),
        inventory_sha256=plan['inventory_sha256'],groups=sources,report_code_sha256=D.Q.sha256(__file__),
        qa_code_sha256=D.Q.sha256(D.__file__),score_rules=dict(
            RF='expected-answer accuracy; one question, one frame; max canonical GT mask area, earliest tie; scene first frame',
            AP='expected-answer accuracy; one joint multi-frame answer per fact',
            EntityCount='exact count accuracy; global proposition scores 1 only when every sampled frame is exact; local remains per-frame; no composite DF',
            SL='AC only, wrong source color yes proportion; lower is better')))
    print(dict(counts),flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','judge','report'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inventory',type=Path)
    p.add_argument('--protocol-dir',type=Path,default=DEFAULT_PROTOCOL)
    p.add_argument('--worker',type=int,default=0)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--model-path')
    p.add_argument('--job-list',type=Path,help='Fixed per-worker job indices for resuming an interrupted run')
    p.add_argument('--snapshot',action='store_true',help='Read live answer checkpoints without rewriting group reports')
    a=p.parse_args();a.out=a.out.resolve()
    if a.action=='prepare':prepare(a.out,a.inventory or a.out/'inventory.json',a.protocol_dir,a.workers)
    elif a.action=='judge':judge(a.out,a.worker,a.workers,a.batch_size,a.model_path,job_list=a.job_list)
    else:report(a.out,a.snapshot)


if __name__=='__main__':main()
