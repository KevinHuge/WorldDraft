"""Export declared event boundaries from final Blender state and published draft time maps."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import subprocess
from pathlib import Path

from worlddraft_eval.common import paths
from worlddraft_eval.gt.visible import repo_rel, sha256, source_to_blender_frame

ROOT = paths.REPO
PROTOCOL = paths.PROTOCOL/'events'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def timestamp_interval(frame, mapping, timestamps):
    """Bracket a source-state transition on the actual published-video sample timeline."""
    if len(mapping)!=len(timestamps) or not mapping or any(b<a for a,b in zip(mapping,mapping[1:])):
        raise ValueError('Invalid source-to-video mapping')
    index=bisect.bisect_left(mapping,frame)
    if index==0 or index==len(mapping):
        return None
    return [timestamps[index-1],timestamps[index]]


def runs(states):
    result=[];start=None
    for index,state in enumerate([*states,False]):
        if state and start is None:start=index
        elif not state and start is not None:
            result.append((start,index-1));start=None
    return result


def boundaries(values, rule):
    """Apply only explicitly declared scalar/directional/visibility criteria."""
    span=max(values)-min(values)
    if not math.isfinite(span):raise ValueError('Nonfinite source state')
    mode=rule['take']
    if mode=='crossing':
        threshold=rule['threshold'];direction=rule['direction']
        transitions=[i for i in range(1,len(values)) if
                     (values[i-1]<threshold<=values[i] if direction>0 else values[i-1]>threshold>=values[i])]
        segments=[(i,i) for i in transitions]
    elif mode=='hidden':
        segments=[s for s in runs([v<=rule['threshold'] for v in values]) if s[1]-s[0]>=2]
    else:
        epsilon=max(1e-7,span*1e-5)
        segments=[(a,b+1) for a,b in runs([(b-a)*rule['direction']>epsilon for a,b in zip(values,values[1:])])
                  if b-a>=2 and abs(values[b+1]-values[a])>=span*.005]
    occurrence=rule['occurrence']
    index=occurrence-1 if occurrence>0 else occurrence
    if not segments or not -len(segments)<=index<len(segments):
        return dict(status='no_reference_event',boundaries={},segments=segments)
    start,end=segments[index]
    if mode=='hidden':end+=1
    elif mode!='crossing':start+=1
    requested={'time':start} if mode=='crossing' else (
        {'onset':start} if mode=='onset' else {'offset':end} if mode=='offset' else {'onset':start,'offset':end})
    # Clip edges cannot establish when an event began or ended outside observation.
    valid={k:v for k,v in requested.items() if 0<v<len(values)-1}
    return dict(status='complete' if valid else 'clipped_reference',boundaries=valid,
                clipped=[k for k in requested if k not in valid],segments=segments)


def pts(video):
    raw=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                                 'frame=best_effort_timestamp_time','-of','json',str(video)],text=True)
    times=[float(f['best_effort_timestamp_time']) for f in json.loads(raw)['frames']]
    times=[t-times[0] for t in times]
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Nonmonotonic published video PTS')
    return times


def target_objects(obj):
    return [o for o in [obj,*obj.children_recursive] if not o.hide_render and o.type=='MESH']


def scene_signal(scene,obj,rule,initial_quaternion):
    signal=rule['signal'];evaluated=obj.evaluated_get(__import__('bpy').context.evaluated_depsgraph_get())
    if signal.startswith('world_'):return float(evaluated.matrix_world.translation['xyz'.index(signal[-1])])
    if signal.startswith('local_'):return float(evaluated.location['xyz'.index(signal[-1])])
    if signal.startswith('rotation_'):return float(evaluated.rotation_euler['xyz'.index(signal[-1])])
    if signal=='tray_gap':
        from mathutils import Vector
        receiver=scene.objects[rule['receiver']]
        corners=[receiver.matrix_world@Vector(v) for v in receiver.bound_box]
        lo=[min(v[i] for v in corners) for i in range(3)]
        hi=[max(v[i] for v in corners) for i in range(3)]
        position=evaluated.matrix_world.translation
        return float(position.z-max(evaluated.dimensions)/2-hi[2]) if lo[0]<=position.x<=hi[0] and lo[1]<=position.y<=hi[1] else 1e6
    if signal=='shape_time':return float(obj.data.shape_keys.eval_time)
    if signal=='displacement':return float((evaluated.matrix_world.translation-initial_quaternion[1]).length)
    if signal=='tilt_deg':return math.degrees(initial_quaternion[0].rotation_difference(evaluated.matrix_world.to_quaternion()).angle)
    if signal=='height':
        from mathutils import Vector
        z=[(evaluated.matrix_world@Vector(v)).z for v in evaluated.bound_box]
        return max(z)-min(z)
    if signal!='visibility':raise ValueError(f'Unknown signal: {signal}')
    import bpy
    from mathutils import Vector
    depsgraph=bpy.context.evaluated_depsgraph_get()
    objects=target_objects(obj);names={o.name for o in objects}
    vertices=[]
    for original in objects:
        target=original.evaluated_get(depsgraph)
        mesh=target.to_mesh()
        try:
            # Deterministic surface support samples, not a raster visibility-area estimate.
            vertices.extend(target.matrix_world@mesh.vertices[i].co for i in range(0,len(mesh.vertices),max(1,len(mesh.vertices)//64)))
        finally:target.to_mesh_clear()
    if not vertices:raise ValueError(f'No renderable target geometry: {obj.name}')
    vertices=vertices[::max(1,len(vertices)//64)][:64]
    visible=0;camera=scene.camera;camera_position=camera.matrix_world.translation
    direction=camera.matrix_world.to_quaternion()@Vector((0,0,-1))
    for point in vertices:
        origin=camera_position if camera.data.type=='PERSP' else point-direction*((point-camera_position).dot(direction))
        ray=point-origin
        if ray.length<=1e-8:continue
        hit,location,normal,index,hit_obj,matrix=scene.ray_cast(depsgraph,origin,ray.normalized(),distance=ray.length+1e-4)
        visible+=bool(hit and hit_obj.name in names)
    return visible/len(vertices)


def projected_box(scene,obj,width,height):
    import bpy
    from mathutils import Vector
    from bpy_extras.object_utils import world_to_camera_view
    depsgraph=bpy.context.evaluated_depsgraph_get();points=[]
    for original in target_objects(obj):
        evaluated=original.evaluated_get(depsgraph)
        points.extend(world_to_camera_view(scene,scene.camera,evaluated.matrix_world@Vector(v)) for v in evaluated.bound_box)
    front=[p for p in points if p.z>0]
    if not front:return None
    box=[max(0,min(p.x for p in front)*width),max(0,(1-max(p.y for p in front))*height),
         min(width,max(p.x for p in front)*width),min(height,(1-min(p.y for p in front))*height)]
    return box if box[2]>box[0] and box[3]>box[1] else None


def export(root=None, protocol=PROTOCOL, out=None):
    root=Path(root or paths.DATA)  # data directory: drafts/, scenes/, gt/
    import bpy
    out=out or root/'gt/events'
    rules=json.loads((protocol/'timing_rules.json').read_text())
    legacy=json.loads((paths.PROTOCOL/'motion-scope.json').read_text())
    registry=dict(schema='worlddraft.event_registry',dataset='worlddraft',sources=[],definitions=[],instances=[],
                  worlddraft=dict(rule_version=rules['rule_version'],timing_rules_sha256=sha256(protocol/'timing_rules.json'),
                           scope_notes=rules['excluded'],numerical_tolerance=rules['numerical_tolerance'],
                           visibility_rule=rules['visibility_rule'],timestamp_rule=rules['timestamp_rule']))
    events=dict(schema='worlddraft.events',dataset='worlddraft',window_pad_seconds=.15,episodes={})
    descriptions=dict(schema='worlddraft.descriptions',dataset='worlddraft',episodes={})
    audit=[]
    for case,spec in legacy['cases'].items():
        descriptions['episodes'][case]={'reference':dict(timeline=[],end_state=spec['end_state'])}
        events['episodes'][case]=[dict(name=e['id'],phases=[p+'#1' for p in e['phases']],participants=e['participants'],
                                          action=e['action'],outcome=e['outcome'],worlddraft=dict(purpose='legacy_event_qa')) for e in spec['events']]
        for phase in spec['timeline']:
            event_id=f"{case}/legacy/{phase['phase']}#1"
            instance=dict(event_id=event_id,definition_id=f"{case}/phase/{phase['phase']}",episode=case,kind='phase',
                          name=phase['phase'],start_time=phase['t0'],end_time=phase['t1'],
                          start_frame=round(phase['t0']*16),end_frame_exclusive=round(phase['t1']*16),
                          participants=[],payload={'worlddraft':dict(gt_method='legacy_manual_phase_annotation',timing_eligible=False)},evidence={})
            registry['instances'].append(instance)
            descriptions['episodes'][case]['reference']['timeline'].append(dict(phase=phase['phase']+'#1',
                startFrame=instance['start_frame'],endFrameExclusive=instance['end_frame_exclusive'],
                startSeconds=instance['start_time'],endSeconds=instance['end_time'],description=phase['text']))
        if case not in rules['rules']:continue
        blend=root/f'scenes/{case}/{case}_P2_AA.blend'
        prep_path=root/f'drafts/{case}/{case}_P2_AA/preprocessing.json';prep=json.loads(prep_path.read_text())
        video=prep_path.parent/f'{case}_P2_AA.mp4'
        if sha256(video)!=prep['draft_sha256']:raise ValueError(f'Reference changed: {case}')
        condition_maps={}
        for polish in ('P0','P1','P2'):
            for alignment in ('AN','AA','AC'):
                cell=f'{polish}_{alignment}';p=root/f'drafts/{case}/{case}_{cell}/preprocessing.json'
                other=json.loads(p.read_text())
                if any(other[k]!=prep[k] for k in ('frame_indices','source_frames','fps','frames')):
                    raise ValueError(f'Condition time mapping differs: {case}/{cell}')
                condition_maps[cell]=dict(path=repo_rel(p),sha256=sha256(p))
        timestamps=pts(video)
        bpy.ops.wm.open_mainfile(filepath=str(blend),use_scripts=False)
        scene=bpy.context.scene
        for hidden in scene.objects:
            if hidden.hide_render:hidden.hide_viewport=True
        mapping=[source_to_blender_frame(i,prep['source_frames'],scene.frame_start,scene.frame_end) for i in prep['frame_indices']]
        if len(mapping)!=len(timestamps):raise ValueError(f'PTS mapping length mismatch: {case}')
        source=dict(episode=case,blend=repo_rel(blend),sha256=sha256(blend),video=repo_rel(video),video_sha256=sha256(video),
                    preprocessing=repo_rel(prep_path),preprocessing_sha256=sha256(prep_path),conditions=condition_maps,
                    frame_start=scene.frame_start,frame_end=scene.frame_end,blender_frame_mapping=mapping,published_pts=timestamps)
        registry['sources'].append(source)
        scene.frame_set(scene.frame_start)
        targets={r['object']:scene.objects.get(r['object']) for r in rules['rules'][case]}
        missing=[name for name,obj in targets.items() if obj is None]
        if missing:raise ValueError(f'{case}: missing rule targets {missing}')
        initial={name:(obj.matrix_world.to_quaternion().copy(),obj.matrix_world.translation.copy()) for name,obj in targets.items()}
        signals={r['name']:[] for r in rules['rules'][case]}
        boxes={r['name']:[] for r in rules['rules'][case]}
        width=round(scene.render.resolution_x*.5);height=round(scene.render.resolution_y*.5)
        for frame in range(scene.frame_start,scene.frame_end+1):
            scene.frame_set(frame)
            for rule in rules['rules'][case]:
                signals[rule['name']].append(scene_signal(scene,targets[rule['object']],rule,initial[rule['object']]))
                boxes[rule['name']].append(projected_box(scene,targets[rule['object']],width,height))
        for rule in rules['rules'][case]:
            signal=signals[rule['name']];found=boundaries(signal,rule);evidence={}
            for boundary,index in found['boundaries'].items():
                frame=scene.frame_start+index;interval=timestamp_interval(frame,mapping,timestamps)
                if interval is not None:
                    evidence[boundary]=dict(blender_frame=frame,interval_s=interval,time_s=sum(interval)/2,
                                             before_value=signal[index-1],value=signal[index],
                                             before_video_frame=bisect.bisect_left(mapping,frame)-1,
                                             after_video_frame=bisect.bisect_left(mapping,frame))
            status='complete' if evidence else found['status'] if found['status']!='complete' else 'clipped_reference'
            name=rule['name'];event_id=f'{case}/{rules["rule_version"]}/{name}#1'
            inst=dict(event_id=event_id,definition_id=f'{case}/timing/{name}',episode=case,kind='state_change' if len(evidence)==1 else 'phase',
                      name=name,occurrence=rule['occurrence'],start_time=min((v['time_s'] for v in evidence.values()),default=None),
                      end_time=max((v['time_s'] for v in evidence.values()),default=None),participants=[{'entity_id':rule['object'],'known_entity':True}],
                      payload={'worlddraft':dict(gt_method='evaluated_state_rule',rule_version=rules['rule_version'],timebase='published_draft_seconds',
                                          status=status,rule=rule,clipped_boundaries=found.get('clipped',[]),required_boundaries=list(evidence))},evidence=evidence)
            registry['instances'].append(inst)
            registry['definitions'].append(dict(definition_id=inst['definition_id'],episode=case,name=name,rule=rule))
            events['episodes'][case].append(dict(name=name,phases=[name+'#1'],participants=[rule['object']],action=rule['description'],
                                                  outcome=rule['description'],worlddraft_timing=dict(rule_version=rules['rule_version'],status=status,
                                                   required_boundaries=list(evidence),parent_event=rule['parent_event'],occurrence=rule['occurrence'])))
            audit.append(dict(case=case,event=name,status=status,boundaries=evidence,segments=found['segments'],
                              signal_min=min(signal),signal_max=max(signal)))
        write_json(out/'source_states'/f'{case}.json',dict(source=source,signals=signals,boxes=boxes,width=prep['source_metadata']['streams'][0]['width'],height=prep['source_metadata']['streams'][0]['height']))
        write_json(out/'registry.partial.json',registry)
        print(case,[(a['event'],a['status']) for a in audit if a['case']==case],flush=True)
    registry['version']=hashlib.sha256(json.dumps(registry,sort_keys=True).encode()).hexdigest()
    write_json(out/'registry.json',registry);write_json(out/'boundary_audit.json',audit)
    write_json(protocol/'events.json',events);write_json(protocol/'descriptions.json',descriptions)
    write_json(protocol/'settings.json',dict(dataset='worlddraft',files=dict(events='events.json',descriptions='descriptions.json',
                registry=repo_rel(out/'registry.json'),legacy_qa=repo_rel(paths.PROTOCOL/'motion-scope.json')),
                judge=dict(model='Qwen3.6-27B',temperature=0,answers='yes/no only for new QA; legacy answers preserved'),
                interaction=legacy['settings'],event_timing=dict(protocol='evaluated-motion-boundaries',
                coarse_frames=11,fine_hz=8,fine_radius_s=.625,missing='no finite timing error; detection coverage reported',
                gt_point='midpoint of preserved GT interval',prediction_point='midpoint of adjacent evidence-frame interval',
                repeated_events='declared target identity and occurrence; never nearest-to-GT matching')))
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=paths.DATA,help='Data directory')
    parser.add_argument('--protocol',type=Path,default=PROTOCOL)
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();export(args.root,args.protocol,args.out)


if __name__=='__main__':main()
