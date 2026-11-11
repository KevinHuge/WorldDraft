"""Blender semantic entities and their projected boxes (data/gt/entities) for the geometry stage."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import re

import numpy as np

from worlddraft_eval.common import paths
from worlddraft_eval.gt.visible import (collect_entities, repo_rel, letterbox_transform, preprocessing_transform, sha256, source_to_blender_frame)
from worlddraft_eval.gt.regions import geometry_bounds
from worlddraft_eval.gt import boxes as P
from worlddraft_eval.geometry.common import write_json

ROOT = paths.REPO
PROTOCOL = paths.PROTOCOL / "entities.json"


def load_protocol(path=PROTOCOL):
    protocol = json.loads(Path(path).read_text())
    if set(protocol["cases"]) != {f"{s}{i:02d}" for s in "ABCDEF" for i in range(1, 9)}:
        raise ValueError("Entity protocol must define all 48 cases")
    for case, rules in protocol["cases"].items():
        if not rules:
            raise ValueError(f"No entity rules for {case}")
        for rule in rules:
            re.compile(rule["root_pattern"])
            if rule["label"] not in protocol["prompts"]:
                raise ValueError(f"Missing category phrase: {case}/{rule['label']}")
    return protocol


def select_entities(scene, case, rules):
    """Semantic roots own their descendants; explicitly merge unparented parts."""
    result, claimed = {}, set()
    for rule in rules:
        selected = collect_entities(scene, case, dict(gt_root_pattern=rule["root_pattern"], attributes=[]))
        groups = [selected] if rule.get("merge_as") else [[e] for e in selected]
        for group in groups:
            eid = rule.get("merge_as") or group[0]["entity_id"]
            objects = {o.name: o for e in group for o in e["objects"]}
            if eid in result or claimed.intersection(objects):
                raise ValueError(f"Overlapping semantic entity: {case}/{eid}")
            claimed.update(objects)
            result[eid] = dict(id=eid, label=rule["label"], dynamic=rule["dynamic"],
                               interaction_candidate=rule["interaction_candidate"],
                               roots=[e["root"].name for e in group], objects=list(objects.values()))
    return result


def map_source_box(box, transform):
    sw, sh = transform["source_size"]
    rw, rh = transform["resized_size"]
    ox, oy = transform["offset_xy"]
    return [box[0] * rw / sw + ox, box[1] * rh / sh + oy,
            box[2] * rw / sw + ox, box[3] * rh / sh + oy]


def projected_box(corners, camera, projection, size, camera_type):
    if camera_type == "PERSP":
        return P.box_from_corners(camera, corners)[0]
    # Blender also supplies orthographic scenes; the upstream engine uses perspective cameras.
    h = np.column_stack([corners, np.ones(len(corners))])
    local = h @ np.asarray(camera["worldToCamera"]).reshape(4, 4)
    if not np.any(local[:, 2] < -1e-3):
        return None
    ndc = local @ projection.T
    xy = (ndc[:, :2] / ndc[:, 3:4] * [1, -1] + 1) * np.asarray(size) / 2
    return [*xy.min(axis=0), *xy.max(axis=0)]


def export_case(case, out, protocol_path=PROTOCOL):
    import bpy

    protocol = load_protocol(protocol_path)
    destination = Path(out) / f"{case}.json"
    if destination.exists():
        raise FileExistsError(destination)
    blend = paths.scene_blend(case)
    prep_path = paths.preprocessing(case)
    prep = json.loads(prep_path.read_text())
    reference = prep_path.parent / f"{case}_P2_AA.mp4"
    if sha256(reference) != prep["draft_sha256"]:
        raise ValueError(f"Reference video changed: {case}")
    if (prep["frames"], prep["fps"], len(prep["frame_indices"])) != (81, 16, 81):
        raise ValueError(f"Unexpected source timeline: {case}")
    source = prep["source_metadata"]["streams"][0]
    size = [int(source["width"]), int(source["height"])]
    canvas = list(map(int, prep["size"].split("*")))
    transform = preprocessing_transform(prep)
    blend_hash = sha256(blend)
    bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    if not scene.camera or scene.camera.data.type not in {"ORTHO", "PERSP"}:
        raise ValueError(f"Unsupported camera: {case}")
    render_size = [int(scene.render.resolution_x * .5), int(scene.render.resolution_y * .5)]
    if [n + n % 2 for n in render_size] != size:
        raise ValueError(f"Source render dimensions changed: {case}")
    entities = select_entities(scene, case, protocol["cases"][case])
    definitions = {e: {**{k: v for k, v in d.items() if k != "objects"},
                       "member_objects": [o.name for o in d["objects"]]} for e, d in entities.items()}
    frames, cache = [], {}
    current_frame = scene.frame_start - 1
    for i, source_index in enumerate(prep["frame_indices"]):
        bf = source_to_blender_frame(source_index, prep["source_frames"], scene.frame_start, scene.frame_end)
        if bf not in cache:
            for frame in range(current_frame + 1, bf + 1):
                scene.frame_set(frame)
            current_frame = bf
            deps = bpy.context.evaluated_depsgraph_get()
            cam = scene.camera.evaluated_get(deps)
            projection = np.asarray(cam.calc_matrix_camera(deps, x=render_size[0], y=render_size[1],
                scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y), dtype=float)
            w, h = render_size
            camera = dict(worldToCamera=np.asarray(cam.matrix_world.inverted()).T.reshape(-1).tolist(),
                          intrinsics=dict(fx=projection[0, 0] * w / 2, fy=projection[1, 1] * h / 2,
                              cx=(1 - projection[0, 2]) * w / 2, cy=(1 + projection[1, 2]) * h / 2))
            boxes = {}
            for eid, entity in entities.items():
                points = []
                for original in entity["objects"]:
                    obj = original.evaluated_get(deps)
                    if obj.hide_render:
                        continue
                    bounds = np.asarray(geometry_bounds(obj))
                    points.extend((np.column_stack([bounds, np.ones(len(bounds))]) @
                                   np.asarray(obj.matrix_world).T)[:, :3])
                box = raw = None
                if points:
                    points = np.asarray(points)
                    corners = np.asarray(list(product(*zip(points.min(0), points.max(0)))))
                    raw = projected_box(corners, camera, projection, render_size, cam.data.type)
                    box = P.clip_box(raw, *size) if raw is not None else None
                visible = bool(box is not None and box[2] - box[0] >= 2 and box[3] - box[1] >= 2)
                boxes[eid] = dict(box=map_source_box(box, transform) if box is not None else None,
                                  visible=visible, source_box=box)
            cache[bf] = boxes
        frames.append(dict(frame=i, time=i / 16, source_frame=source_index, blender_frame=bf, boxes=cache[bf]))
    if sha256(blend) != blend_hash:
        raise RuntimeError(f"Source blend changed: {blend}")
    result = dict(schema="worlddraft.gt", episode=case, camera="blender", fps=16, frame_count=81,
        width=canvas[0], height=canvas[1], entities=definitions, frames=frames,
        source=dict(blend=repo_rel(blend), blend_sha256=blend_hash, preprocessing=repo_rel(prep_path),
                    preprocessing_sha256=sha256(prep_path), reference=repo_rel(reference),
                    reference_sha256=prep["draft_sha256"], protocol_sha256=sha256(Path(protocol_path)),
                    exporter_sha256=sha256(Path(__file__)), source_blend_unchanged=True),
        adaptation=dict(projection="Projected world AABB of semantic entity geometry, clipped to source canvas; no occlusion test, matching upstream amodal-box convention.",
                        source_fps=16, source_size=size, letterbox=transform, camera_type=scene.camera.data.type,
                        visibility="render-visible geometry; projected source box width and height >= 2 px",
                        entity_selection="Explicit semantic entities, dynamic or interaction candidates; no color/material prompts."))
    write_json(destination, result)
    print(f"{case}: {len(entities)} entities, {len(frames)} frames -> {destination}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    for case in args.cases or load_protocol(args.protocol)["cases"]:
        export_case(case, args.out, args.protocol)


if __name__ == "__main__":
    main()
