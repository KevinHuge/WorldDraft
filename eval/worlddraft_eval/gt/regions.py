"""Project interaction participants on all 81 reference-video instants.

CPU-only Blender export. Boxes select evidence regions, not entity/IoU scores.
"""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path

from worlddraft_eval.common import paths
from worlddraft_eval.gt.visible import collect_entities, repo_rel, sha256, source_to_blender_frame

ROOT = paths.REPO
PROTOCOL = paths.PROTOCOL / "motion-scope.json"


def geometry_bounds(obj):
    if obj.type == "MESH":
        return list(obj.bound_box)
    # Curve bound_box can include an unrendered unit-radius envelope.
    mesh = obj.to_mesh()
    try:
        if mesh is None or not mesh.vertices:
            raise ValueError(f"No evaluated geometry: {obj.name}")
        coords = [tuple(v.co) for v in mesh.vertices]
        return list(product(*[(min(v[i] for v in coords), max(v[i] for v in coords)) for i in range(3)]))
    finally:
        obj.to_mesh_clear()


def project_box(points, width, height):
    """Clip projected geometry to the canvas; an offscreen group has no box."""
    front = [p for p in points if p[2] > 0]
    if not front:
        return None
    x0 = max(0, min(p[0] for p in front) * width)
    x1 = min(width, max(p[0] for p in front) * width)
    y0 = max(0, (1 - max(p[1] for p in front)) * height)
    y1 = min(height, (1 - min(p[1] for p in front)) * height)
    if x1 <= x0 or y1 <= y0:
        return None
    if len(front) != len(points):
        raise ValueError("Visible participant bounds cross the camera plane")
    return [x0, y0, x1, y1]


def export_case(case, protocol, out):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    destination = Path(out) / f"{case}.json"
    if destination.exists():
        raise FileExistsError(destination)
    spec = protocol["cases"][case]
    blend = paths.scene_blend(case)
    prep_path = paths.preprocessing(case)
    prep = json.loads(prep_path.read_text())
    reference = prep_path.parent / f"{case}_P2_AA.mp4"
    if sha256(reference) != prep["draft_sha256"]:
        raise ValueError(f"{case}: reference video differs from preprocessing manifest")
    if prep["frames"] != 81 or prep["fps"] != 16 or len(prep["frame_indices"]) != 81:
        raise ValueError(f"{case}: reference timeline changed")
    blend_hash = sha256(blend)
    bpy.ops.wm.open_mainfile(filepath=str(blend), use_scripts=False)
    scene = bpy.context.scene
    if not scene.camera or scene.camera.data.type not in {"ORTHO", "PERSP"}:
        raise ValueError("A perspective or orthographic camera is required")
    source = prep["source_metadata"]["streams"][0]
    width, height = source["width"], source["height"]
    # Source MP4 may pad an odd native half-resolution by one encoder pixel.
    content_w = round(scene.render.resolution_x * .5)
    content_h = round(scene.render.resolution_y * .5)
    if width - content_w not in (0, 1) or height - content_h not in (0, 1):
        raise ValueError(f"{case}: source canvas mapping changed")
    groups, selection = {}, {}
    for name, pattern in spec["groups"].items():
        entities = collect_entities(scene, case, {"gt_root_pattern": pattern, "attributes": []})
        roots = [e["root"] for e in entities]
        members = {o.name: o for e in entities for o in e["objects"] if not o.hide_render}
        if not roots or not members:
            raise ValueError(f"{case}/{name}: no renderable participants match {pattern}")
        groups[name] = list(members.values())
        selection[name] = {"pattern": pattern, "roots": [r.name for r in roots],
                           "members": sorted(members)}
    frames, cached = [], {}
    for i, source_index in enumerate(prep["frame_indices"]):
        bf = source_to_blender_frame(source_index, prep["source_frames"],
                                    scene.frame_start, scene.frame_end)
        if bf not in cached:
            scene.frame_set(bf)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            boxes = {}
            for name, objects in groups.items():
                visible = []
                for original in objects:
                    obj = original.evaluated_get(depsgraph)
                    points = [tuple(world_to_camera_view(scene, scene.camera,
                              obj.matrix_world @ Vector(v))) for v in geometry_bounds(obj)]
                    box = project_box(points, content_w, content_h)
                    if box:
                        visible.append(box)
                box = ([min(b[0] for b in visible), min(b[1] for b in visible),
                        max(b[2] for b in visible), max(b[3] for b in visible)] if visible else None)
                boxes[name] = {"visible": box is not None, "box": box}
            cached[bf] = boxes
        frames.append({"frame": i, "time": i / prep["fps"], "source_frame": source_index,
                       "blender_frame": bf, "boxes": cached[bf]})
    if sha256(blend) != blend_hash:
        raise RuntimeError(f"Source blend changed: {blend}")
    result = {"schema": "worlddraft.motion-region-gt", "case": case, "width": width, "height": height,
              "fps": prep["fps"], "frames": frames, "groups": selection,
              "source_blend": repo_rel(blend), "source_blend_sha256": blend_hash,
              "reference_video": repo_rel(reference), "reference_sha256": sha256(reference),
              "preprocessing_sha256": sha256(prep_path),
              "projection": "Amodal evaluated-mesh bounding boxes, clipped to source canvas; no occlusion masks.",
              "purpose": "Interaction-region zoom only; visible means intersects canvas, not visually unobscured.",
              "source_blend_unchanged": True, "code_sha256": sha256(Path(__file__))}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"{case}: {len(groups)} groups, {len(frames)} instants -> {destination}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    for case in args.cases or protocol["cases"]:
        export_case(case, protocol, args.out)


if __name__ == "__main__":
    main()
