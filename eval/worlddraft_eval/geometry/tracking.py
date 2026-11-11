"""Entity tracking on generated videos: SAM 3 text-prompted detection and tracking, completed by SAM 2.1 seeded with
ground-truth boxes for categories SAM 3 misses. Writes worlddraft.tracks files."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from worlddraft_eval.geometry.common import (EVAL_DURATION, GeneratedSample, eval_grid, evaluated_entities, iou_matrix, output_frame_for_time, source_frame_for_time, video_meta, write_json, iter_video_frames)

LABEL_QUERY = {
    "person": "person", "crate": "wooden crate", "parcel": "wooden box", "ball": "ball", "bench": "bench",
    "chair": "chair", "platform": "platform", "tree": "tree", "building": "building", "car": "car", "horse": "horse",
    "boat": "boat", "zipline": "zipline cable", "cable": "zipline cable", "handle": "zipline trolley handle",
    "jetpack": "jetpack",
    "bow": "bow", "arrow": "arrow", "target": "archery target", "door": "door",
    "vehicle": "vehicle", "cart": "handcart", "obstacle": "fallen log", "control-panel": "control panel switch",
    "checkpoint": "start gate arch", "barrier": "guard rail", "road-marking": "road marking",
    "ladder": "ladder", "weapon": "weapon", "fishing-rod": "fishing rod", "fishing-line": "fishing line", "fish": "fish",
    "machine": "arcade claw machine", "claw": "claw", "prize": "plush toy", "tool": "comb", "dart": "dart", "cup": "cup", "table": "table", "sculpture": "statue",
    "dog": "dog", "cat": "cat", "rabbit": "rabbit", "bird": "bird", "duck": "duck", "animal": "animal",
}


def queries_from_gt(gt: dict, include_static: bool = False) -> list[str]:
    """Detector phrases for the evaluated entities (never terrain / effect / geometry). Raises on an unmapped
    label so a new episode cannot silently query the raw label."""
    labs = []
    for e in evaluated_entities(gt, "all" if include_static else "dynamic_or_candidate"):
        lab = gt["entities"][e]["label"]
        if lab not in LABEL_QUERY:
            raise KeyError(f"GT label {lab!r} (entity {e}) has no detector phrase in LABEL_QUERY")
        q = LABEL_QUERY[lab]
        if q not in labs:
            labs.append(q)
    return labs


def _load_detector(model: str, device: str):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(model)
    net = AutoModelForZeroShotObjectDetection.from_pretrained(model).to(device).eval()
    return proc, net, torch


def detect_batch(proc, net, torch, images, text: str, box_thresh: float, text_thresh: float, device: str):
    from PIL import Image
    pil = [Image.fromarray(im[:, :, ::-1]) for im in images]
    inputs = proc(images=pil, text=[text] * len(pil), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = net(**inputs)
    sizes = torch.tensor([[im.shape[0], im.shape[1]] for im in images], device=device)
    try:
        res = proc.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=box_thresh,
                                                          text_threshold=text_thresh, target_sizes=sizes)
    except TypeError:
        res = proc.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=box_thresh,
                                                          text_threshold=text_thresh, target_sizes=sizes)
    out = []
    for r in res:
        labels = r.get("text_labels") or r.get("labels")
        out.append((r["boxes"].cpu().numpy(), [str(name) for name in labels], r["scores"].cpu().numpy()))
    return out


def _load_sam2(model: str, device: str):
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    net = Sam2VideoModel.from_pretrained(model, dtype=torch.bfloat16).to(device).eval()
    proc = Sam2VideoProcessor.from_pretrained(model)
    return proc, net, torch


def mask_to_box(mask, min_pixels: int = 4):
    """Tight pixel AABB [x0,y0,x1,y1] of a binary mask, or None if fewer than min_pixels are set."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _sam2_propagate(proc, net, torch, frames, gt, ent_ids, seeds, n_frames, meta, device, min_pixels, non_overlapping,
                    reseed_every, reseed_min_iou, detector, init_model, entities):
    """SAM2 propagation for the entities in `seeds` (entity -> dict(frame, box, ...)); returns (dets per frame,
    reseed log). track_id = 1-based index of the entity in ent_ids. Used by run_sam2 and by run_sam3's fallback."""
    dets = {i: [] for i in range(n_frames)}
    groups = {}
    for e, v in seeds.items():
        groups.setdefault(v["frame"], []).append((e, v["box"]))
    reseed_frames = []
    if reseed_every and reseed_every > 0:
        step = max(1, int(round(reseed_every * meta["fps"])))
        reseed_frames = list(range(step, n_frames, step))
    reseed_log = []
    if reseed_frames and detector is None:
        detector = _load_detector(init_model, device)
    if reseed_frames:
        dproc, dnet, _ = detector
        queries = queries_from_gt(gt, include_static=(entities == "all"))
        text = ". ".join(queries) + "."
        from worlddraft_eval.geometry.entity_metrics import DEFAULT_LABEL_MAP, _label_ok

    def _collect(o, sess, box_now):
        m = proc.post_process_masks([o.pred_masks], original_sizes=[[sess.video_height, sess.video_width]],
                                    binarize=True, apply_non_overlapping_constraints=non_overlapping)[0]
        for i, oid in enumerate(sess.obj_ids):
            mk = m[i, 0].cpu().numpy().astype(bool)
            box = mask_to_box(mk, min_pixels)
            if box is None:
                continue
            e = ent_ids[oid - 1]
            box_now[oid] = box
            dets[o.frame_idx].append(dict(track_id=int(oid), label=gt["entities"][e]["label"], entity=e,
                                          score=float(mk.mean()), box=box))

    for f0, members in sorted(groups.items()):
        sess = proc.init_video_session(video=frames, inference_device=device, dtype=torch.bfloat16)
        obj_ids = [ent_ids.index(e) + 1 for e, _ in members]
        proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=f0, obj_ids=obj_ids,
                                             input_boxes=[[box for _, box in members]])
        box_now = {}
        boundaries = [f for f in reseed_frames if f > f0] + [n_frames]
        with torch.inference_mode():
            _collect(net(inference_session=sess, frame_idx=f0), sess, box_now)
            cur = f0
            for nxt in boundaries:
                for o in net.propagate_in_video_iterator(sess, start_frame_idx=cur, max_frame_num_to_track=nxt - cur):
                    if o.frame_idx == cur:
                        continue
                    _collect(o, sess, box_now)
                if nxt >= n_frames:
                    break
                boxes, labels, scores = detect_batch(dproc, dnet, torch, [frames[nxt][:, :, ::-1]], text, 0.3, 0.25, device)[0]
                labels = [_canon(name, queries) for name in labels]
                new_ids, new_boxes = [], []
                for oid in sess.obj_ids:
                    e = ent_ids[oid - 1]; lab = gt["entities"][e]["label"]; B = box_now.get(oid)
                    if B is None:
                        continue
                    best, best_iou = None, 0.0
                    for bx, dl in zip(boxes, labels):
                        if dl is None or not _label_ok(lab, dl, DEFAULT_LABEL_MAP):
                            continue
                        i = iou_matrix([B], [bx])[0, 0]
                        if i > best_iou:
                            best, best_iou = [float(x) for x in bx], float(i)
                    if best is not None and best_iou >= reseed_min_iou:
                        new_ids.append(int(oid)); new_boxes.append(best)
                        reseed_log.append(dict(frame=nxt, entity=e, iou_with_track=round(best_iou, 3)))
                if new_ids:
                    dets[nxt] = [d for d in dets[nxt] if d["track_id"] not in new_ids]
                    proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=nxt, obj_ids=new_ids,
                                                         input_boxes=[new_boxes])
                    _collect(net(inference_session=sess, frame_idx=nxt), sess, box_now)
                cur = nxt
        del sess
        torch.cuda.empty_cache()
    return dets, reseed_log


def _load_sam3(model: str, device: str, dtype: str = "bfloat16"):
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    net = Sam3VideoModel.from_pretrained(model, dtype=getattr(torch, dtype)).to(device).eval()
    proc = Sam3VideoProcessor.from_pretrained(model)
    return proc, net, torch


SAM3_PROMPT = {}


def run_sam3(video: str, gt: dict, sample: GeneratedSample, out: str, model: str, device: str = "cuda:0",
             grid_hz: float = 8.0, entities: str = "dynamic_or_candidate", tracker=None, min_pixels: int = 4,
             dtype: str = "bfloat16", score_thresh: float | None = None, prompt_overrides: dict | None = None,
             fallback: str = "none", sam2_model: str | None = None, sam2=None, frames_mode: str = "all"):
    """Detect and track the entities of one video with SAM 3 text prompts (optional SAM 2.1 fallback); writes
    worlddraft.tracks to `out`."""
    import numpy as np
    proc, net, torch = tracker or _load_sam3(model, device, dtype)
    meta = video_meta(video)
    n_frames = min(meta["frame_count"], int(round(EVAL_DURATION * meta["fps"])))
    if frames_mode == "all":
        frame_ids = list(range(n_frames))
    elif frames_mode == "grid":
        if fallback != "none":
            raise NotImplementedError("sam2 fallback is not supported with frames_mode='grid'")
        frame_ids = sorted({min(output_frame_for_time(t, meta["fps"]), n_frames - 1) for t in eval_grid(grid_hz)})
    else:
        raise ValueError(f"Unknown frames_mode {frames_mode!r}")
    frames = [im[:, :, ::-1].copy() for _, im in iter_video_frames(video, set(frame_ids))]
    if len(frames) != len(frame_ids):
        raise ValueError(f"Video decode returned {len(frames)} of {len(frame_ids)} requested frames: {video}")
    ent_ids = evaluated_entities(gt, entities)
    labels = []
    for e in ent_ids:
        lab = gt["entities"][e]["label"]
        if lab not in LABEL_QUERY:
            raise KeyError(f"GT label {lab!r} (entity {e}) has no prompt phrase in LABEL_QUERY")
        if lab not in labels:
            labels.append(lab)
    pov = dict(SAM3_PROMPT); pov.update(prompt_overrides or {})
    prompts = {pov.get(name, LABEL_QUERY[name]): name for name in labels}
    if score_thresh is not None:
        net.score_threshold_detection = float(score_thresh)
    sess = proc.init_video_session(video=frames, inference_device=device, processing_device="cpu",
                                   video_storage_device="cpu", dtype=getattr(torch, dtype))
    sess = proc.add_text_prompt(inference_session=sess, text=list(prompts))
    n_fed = len(frame_ids)
    dets = {i: [] for i in range(n_fed)}
    obj_label = {}
    with torch.inference_mode():
        for o in net.propagate_in_video_iterator(inference_session=sess, start_frame_idx=0, max_frame_num_to_track=n_fed):
            po = proc.postprocess_outputs(sess, o)
            if o.frame_idx >= n_fed:
                continue
            p2o = po.get("prompt_to_obj_ids") or {}
            for phrase, ids in p2o.items():
                for oid in ids:
                    obj_label.setdefault(int(oid), prompts.get(phrase, phrase))
            ids = po["object_ids"].tolist() if hasattr(po["object_ids"], "tolist") else list(po["object_ids"])
            for i, oid in enumerate(ids):
                mk = po["masks"][i]
                mk = mk.cpu().numpy().astype(bool) if hasattr(mk, "cpu") else np.asarray(mk, bool)
                box = mask_to_box(mk, min_pixels)
                if box is None:
                    continue
                sc = po["scores"][i]; sc = float(sc.item() if hasattr(sc, "item") else sc)
                dets[o.frame_idx].append(dict(track_id=int(oid) + 1, label=obj_label.get(int(oid), "?"), score=sc, box=box))
    del sess
    torch.cuda.empty_cache()
    fb_info = {}
    if fallback not in ("none", "sam2", "sam2_unmatched"):
        raise ValueError(f"Unknown sam3 fallback {fallback!r}")
    unmatched = []
    if fallback in ("sam2", "sam2_unmatched"):
        found_labels = set(obj_label.values())
        missing_labels = [name for name in labels if name not in found_labels]
        fb_ents = [e for e in ent_ids if gt["entities"][e]["label"] in missing_labels]
        if fallback == "sam2_unmatched":
            from worlddraft_eval.geometry.entity_metrics import evaluate as _entity_eval
            pre = dict(meta, frames=[dict(frame=int(frame_ids[i]), detections=dets[i]) for i in range(n_fed)])
            asg = _entity_eval(gt, pre, sample, grid_hz=grid_hz, entities=entities)["assignment"]
            unmatched = [e for e in ent_ids if e not in asg and e not in fb_ents]
            fb_ents = fb_ents + unmatched
        if fb_ents:
            sp = sample.spatial; W, H = meta["width"], meta["height"]
            seeds = {}
            for e in fb_ents:
                for t in eval_grid(grid_hz):
                    b = gt["frames"][source_frame_for_time(t)]["boxes"].get(e)
                    if not (b and b["visible"]):
                        continue
                    nb = sp.box_src_to_out_norm(b["box"])
                    if (nb[2] - nb[0]) * (nb[3] - nb[1]) < 1e-4:
                        continue
                    of = min(output_frame_for_time(t, meta["fps"]), n_frames - 1)
                    gb = [float(nb[0] * W), float(nb[1] * H), float(nb[2] * W), float(nb[3] * H)]
                    seeds[e] = dict(frame=of, gt_box=gb, box=gb, source="gt", init_iou=None)
                    break
            s2proc, s2net, _ = sam2 or _load_sam2(sam2_model, device)
            dets2, _ = _sam2_propagate(s2proc, s2net, torch, frames, gt, ent_ids, seeds, n_frames, meta, device, min_pixels,
                                       False, 0.0, 0.3, None, "IDEA-Research/grounding-dino-tiny", entities)
            for i in range(n_frames):
                for d in dets2[i]:
                    dets[i].append(dict(track_id=d["track_id"] + 1000, label=d["label"], entity=d["entity"], score=d["score"],
                                        box=d["box"], source="sam2_gtbox"))
            fb_info = dict(labels=missing_labels, entities=fb_ents, unmatched=unmatched, seeds=seeds, sam2_model=sam2_model)
    results = [dict(frame=int(frame_ids[i]), detections=dets[i]) for i in range(n_fed)]
    cfg = getattr(net, "config", None)
    cfg_note = {k: getattr(cfg, k) for k in ("score_threshold_detection", "det_nms_thresh", "hotstart_delay",
                                              "hotstart_unmatch_thresh", "max_num_objects", "high_conf_thresh") if hasattr(cfg, k)}
    payload = dict(schema="worlddraft.tracks", video=str(Path(video).resolve()), **meta, queries=list(prompts),
                   tracker=dict(name="sam3_text", model=model, dtype=dtype, prompts=prompts, grid_hz=grid_hz,
                                frames_mode=frames_mode, n_frames_fed=n_fed,
                                min_pixels=min_pixels, config=cfg_note, score_thresh=net.score_threshold_detection,
                                n_objects=len(obj_label), fallback=fallback, fallback_used=fb_info),
                   frames=results)
    write_json(out, payload)
    n = sum(len(r["detections"]) for r in results)
    print(f"{video}: sam3 {len(results)} frames, {n} boxes, {len(obj_label)} objects for prompts {list(prompts)}"
          + (f", sam2 fallback for {fb_info['entities']}" if fb_info else "") + f" -> {out}")
    return payload


def _canon(label: str, queries: list[str]):
    """Map detector phrase back to one of our queries (GroundingDINO may merge/split words)."""
    phrase = label.lower().strip()
    if not phrase:
        return None
    for q in queries:
        if phrase == q or q in phrase or phrase in q:
            return q
    return None


