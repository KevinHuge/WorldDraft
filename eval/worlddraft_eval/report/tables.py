"""Report tables: selected metrics on families shared by all generators, and paired P×A contrasts."""
import csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
from statistics import mean
from worlddraft_eval.common.paths import record
from worlddraft_eval.report import results as P
from worlddraft_eval.common.metrics import ENTITY_METRICS, MOTION_METRICS, CAMERA_METRICS, RAW
from worlddraft_eval.common import model_config

BASE=P.ROOT/'eval/results/benchmark/report'  # aggregate.py points this at its --out
OUT=BASE/'tables'
REFERENCE=model_config.reference()  # the draft: shown on the generators' shared families, never ranked
CASES=[f'{s}{i:02}' for s in 'ABCDEF' for i in range(1,9)]
CELLS=[f'P{p}_{a}' for p in range(3) for a in ['AN','AA','AC']]
DF=ENTITY_METRICS+MOTION_METRICS+CAMERA_METRICS
RF=['RF_total']+[f'RF_{a}' for a in ['color','material','texture','environment','lighting']]
AP=['AP_total','AP_color','AP_material','AP_texture']
QUALITY=['image_quality','aesthetic_quality','clip_text_visual_alignment']
METRICS=DF+RF+QUALITY+AP+['SL_color']
LABELS={'DF_count_global_accuracy':'Global Count↑','DF_count_local_accuracy':'Local Count↑','bbox_iou':'BBox IoU↑','track_matched_coverage':'Track Coverage↑','ISR':'ISR↑','EventTimingCoverage':'Timing Coverage↑','RotErr_deg_mean':'ViPE Rot(°)↓','TransErr_rel_mean':'ViPE Trans↓','CamMC_rel_mean':'ViPE CamMC↓','RF_total':'RF↑','AP_total':'AP↑','SL_color':'SL Color↓','image_quality':'Image Quality↑','aesthetic_quality':'Aesthetic↑','clip_text_visual_alignment':'CLIP↑'}
SOURCES=set()
def generators(values):
    """Generators present in the results, in eval/configs/models.yaml order."""
    present={k[0] for k in values}
    return [m for m in model_config.models() if m in present and m!=REFERENCE]

def common_color_ids(questions, metadata, bindings):
    """Use audited SL subjects/targets to identify color conflicts shared by all P."""
    sl = [q for q in questions if q['metric'] == 'SL']
    negatives = {metadata[q['proposition_id']]['paired_with']: q['proposition_id']
                 for q in questions if q['metric'] == 'RF' and q['expected_answer'] == 'no'}
    result = {}
    for q in questions:
        if q['metric'] != 'RF' or q['attribute'] != 'color' or q['expected_answer'] != 'yes':
            continue
        pid = q['proposition_id']
        qm = metadata[pid]
        matches = []
        for s in sl:
            if s['entity_id'] != q['entity_id']:
                continue
            sm = metadata[s['proposition_id']]
            if q['item_id'] == 'A01':
                # A01's frozen provenance uses a CSV pointer; its tuple names the body part.
                same = q['tuple'].split(',')[1:-1] == s['tuple'].split(',')[1:-1]
            else:
                same = (qm['subject'], qm['target']) == (sm['subject'], sm['target'])
            if same:
                matches.append(s)
        if all(any(f'P{p}_AC' in bindings.get(s['proposition_id'], {}).get(
                'cells', ['P0_AC', 'P1_AC', 'P2_AC']) for s in matches) for p in range(3)):
            if pid not in negatives:
                raise ValueError(f'Unpaired positive RF: {q}')
            result[pid] = dict(negative=negatives[pid], source_ids=[s['proposition_id'] for s in matches])
    return result


def load(path):
    SOURCES.add(path);return json.loads(path.read_text())

def read(path):
    SOURCES.add(path)
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write(name,rows):
    if not rows:return
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def scale(metric):return 1 if metric in RAW else 100

def cohort(values,models,metric,cells):
    return [c for c in CASES if all((m,c,cell,metric) in values for m in models for cell in cells)]

def summarize(values,models,metric,cells,contrast=None,reference=None):
    """Scores of `models` on their shared complete families; `reference` is scored on the same families when complete."""
    cases=cohort(values,models,metric,cells)
    result={}
    for model in models+([reference] if reference else []):
        if not cases or not all((model,c,cell,metric) in values for c in cases for cell in cells):
            result[model]=None;continue
        vals=[]
        for case in cases:
            v={cell:values[model,case,cell,metric] for cell in cells}
            vals.append(v[contrast[1]]-v[contrast[0]] if contrast else mean(v.values()))
        result[model]=mean(vals)*scale(metric)
    return result,len(cases)

def qa_dir(model,case):return P.case_dir(model,case)

def common_conflict(MODELS):
    records=[];scopes=[];cells=[f'P{p}_{a}' for p in range(3) for a in ['AA','AC']]
    for case in CASES:
        inputs=qa_dir('Wan2.1-VACE-14B',case)/'inputs'
        questions=[q for q in read(inputs/'questions.csv') if q['item_id']==case]
        candidates=common_color_ids(questions,load(inputs/'question_provenance.json')['questions'][case],load(inputs/'settings.json')['qa']['bindings'][case])
        if not candidates:continue
        answers={m:{cell:{r['proposition_id']:r for r in load(qa_dir(m,case)/cell/'answers.json') if r['metric']=='RF'} for cell in cells} for m in MODELS}
        ids=[]
        for pid,info in candidates.items():
            pair=[pid,info['negative']]
            if all(answers[m][cell].get(q,{}).get('eligible') and (answers[m][cell][q]['raw_answer'] or '').strip().lower().rstrip('.!。！').strip() in ['yes','no'] for m in MODELS for cell in cells for q in pair):ids.extend(pair)
        if not ids:continue
        scopes.append(dict(case=case,n_questions=len(ids),question_ids=json.dumps(ids),cells=','.join(cells),models=','.join(MODELS)))
        for m in MODELS:
            v={cell:mean(int(answers[m][cell][q]['raw_answer'].strip().lower().rstrip('.!。！').strip()==answers[m][cell][q]['expected']) for q in ids) for cell in cells}
            cp=[v[f'P{p}_AA']-v[f'P{p}_AC'] for p in range(3)]
            records.append(dict(model=m,case=case,CP_P0=cp[0]*100,CP_P1=cp[1]*100,CP_P2=cp[2]*100,deltaCP_01=(cp[1]-cp[0])*100,deltaCP_12=(cp[2]-cp[1])*100))
    return records,scopes

def level(values,model,cases,cell,metric):
    """Mean over `cases` of one cell, or None when the model lacks any of them."""
    if not cases or not all((model,c,cell,metric) in values for c in cases):return None
    return mean(values[model,c,cell,metric] for c in cases)*scale(metric)

def main():
    OUT.mkdir(parents=True,exist_ok=True);values=P.collect();records=[]
    write('per_video.csv',[dict(model=m,case=c,cell=cell,metric=k,value=v,status='complete') for (m,c,cell,k),v in sorted(values.items())])
    SOURCES.update(P.sources())
    GEN=generators(values);ROWS=GEN+[REFERENCE]
    def add(metric,analysis,condition,cells,contrast=None):
        scores,n=summarize(values,GEN,metric,cells,contrast,REFERENCE)
        for m in ROWS:records.append(dict(analysis=analysis,condition=condition,model=m,metric=metric,value=scores[m],families=n,videos_per_model=n*len(cells),unit='native' if metric in RAW else 'pp' if contrast else 'percent',status='common_complete_subset' if n else 'unavailable',cells=','.join(cells)))
        return scores,n
    for name,cells0 in [('canonical_P1_AN',['P1_AN']),('full_grid',CELLS)]:
        wide={m:{'Model':m} for m in ROWS};coverage=[]
        for metric in METRICS:
            cells=[c for c in cells0 if c.endswith('AC')] if metric=='SL_color' else cells0
            if not cells:
                for m in ROWS:wide[m][metric]=None
                continue
            scores,n=add(metric,'Q1',name,cells)
            for m in ROWS:wide[m][metric]=scores[m]
            coverage.append(dict(metric=metric,families=n,videos_per_model=n*len(cells)))
        write(name+'.csv',list(wide.values()));write(name+'_coverage.csv',coverage)
    for metric in METRICS:
        cells=[c for c in CELLS if metric!='SL_color' or c.endswith('AC')]
        cases=cohort(values,GEN,metric,cells)
        wide=[]
        for m in ROWS:
            row={'Model':m,'families':len(cases)}
            for cell in CELLS:row[cell]=level(values,m,cases,cell,metric) if cell in cells else None
            wide.append(row)
        write('grid_'+metric+'.csv',wide)
    def delta(x,y):return None if x is None or y is None else x-y
    for a in ['AN','AA','AC']:
        for metric in DF+RF+AP+QUALITY:
            cells=[f'P{p}_{a}' for p in range(3)];cases=cohort(values,GEN,metric,cells);wide=[]
            for m in ROWS:
                levels=[level(values,m,cases,cell,metric) for cell in cells]
                row=dict(Model=m,families=len(cases),P0=levels[0],P1=levels[1],P2=levels[2],delta01=delta(levels[1],levels[0]),delta12=delta(levels[2],levels[1]));wide.append(row)
                for label,v in list(row.items())[2:]:records.append(dict(analysis='Q2',condition=a,model=m,metric=metric+':'+label,value=v,families=len(cases),videos_per_model=len(cases)*3,unit='native' if metric in RAW else 'pp' if label.startswith('delta') else 'percent',status='common_complete_triplets' if cases else 'unavailable',cells=','.join(cells)))
    for p in range(3):
        for metric in DF+RF+AP:
            cells=[f'P{p}_{a}' for a in ['AN','AA','AC']];cases=cohort(values,GEN,metric,cells);wide=[]
            for m in ROWS:
                levels=[level(values,m,cases,cell,metric) for cell in cells]
                row=dict(Model=m,families=len(cases),AN=levels[0],AA=levels[1],AC=levels[2],AA_minus_AC=delta(levels[1],levels[2]));wide.append(row)
                for label,v in list(row.items())[2:]:records.append(dict(analysis='Q3',condition=f'P{p}',model=m,metric=metric+':'+label,value=v,families=len(cases),videos_per_model=len(cases)*3,unit='native' if metric in RAW else 'pp' if label=='AA_minus_AC' else 'percent',status='common_complete_triplets' if cases else 'unavailable',cells=','.join(cells)))
    common,scopes=common_conflict(ROWS);write('common_color_question_scope.csv',scopes);write('common_color_per_family.csv',common)
    leak=[]
    for m in ROWS:leak.append({'Model':m})
    for p in range(3):
        scores,n=add('SL_color','Q3',f'P{p}_AC',[f'P{p}_AC'])
        for row in leak:row[f'SL_P{p}_AC']=scores[row['Model']];row[f'n_P{p}']=n
    write('source_leakage_by_polish.csv',leak)
    wide=[]
    for m in ROWS:
        own=[r for r in common if r['model']==m];wide.append(dict(Model=m,families=len(own),**{k:mean(r[k] for r in own) if own else None for k in ['CP_P0','CP_P1','CP_P2','deltaCP_01','deltaCP_12']}))
    write('common_color_conflict.csv',wide)
    write('aggregation_long.csv',records)
    SOURCES.update([Path(__file__),Path(P.__file__)])
    (OUT/'provenance.json').write_text(json.dumps(dict(generated_at=datetime.now(timezone.utc).isoformat(),generators=GEN,reference=REFERENCE,aggregation='cells equal within a family, families equal; each metric on the families complete for every generator; the reference on the same families',source_sha256={record(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)},limitations=['coverage differs per metric (see *_coverage.csv)','timing invalid answers are retried twice, remaining errors stay in the denominator','no composite world-fidelity score','common conflict colors only']),indent=2))
    import zipfile
    with zipfile.ZipFile(OUT/'tables.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.iterdir()):
            if path.suffix in ['.csv','.md','.html','.json']:z.write(path,path.name)
    print(json.dumps(dict(output=str(OUT),tables=len(list(OUT.glob("*.csv"))),aggregation_rows=len(records),common_color_families=len(scopes))))
if __name__=='__main__':main()
