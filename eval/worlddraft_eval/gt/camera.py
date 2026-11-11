"""Export D-series Blender cameras on the saved P2_AA input-video timeline.

Run with the CPU bpy environment. These poses document the authored camera and
provide fixed intrinsics; the experiment's scoring reference remains ViPE's
estimate of the saved draft P2_AA video. No rendering or scene saving is done.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

from worlddraft_eval.common import paths
from worlddraft_eval.gt.visible import letterbox_transform, repo_rel, sha256, source_to_blender_frame


ROOT = paths.REPO
CASES = [f"D{i:02d}" for i in range(1, 9)]
DYNAMIC_PHRASES = {
    "D01": ["person", "car"],
    "D02": ["person", "water tank drum"],
    "D03": ["person"],
    "D04": ["person", "wind turbine rotor"],
    "D05": ["person", "boat"],
    "D06": ["person"],
    "D07": ["person", "train"],
    "D08": ["person", "Ferris wheel gondola", "roller coaster car", "drop tower seat"],
}


def intrinsics_from_projection(projection, native_size, source_size, target_size):
    """OpenGL projection -> top-left pixel K, preserving the source letterbox."""
    native_w, native_h = native_size
    source_w, source_h = source_size
    transform = letterbox_transform(source_size, target_size)
    content_w, content_h = transform["resized_size"]
    offset_x, offset_y = transform["offset_xy"]
    native = {"fx": projection[0][0] * native_w / 2,
              "fy": projection[1][1] * native_h / 2,
              "cx": (1 - projection[0][2]) * native_w / 2,
              "cy": (1 + projection[1][2]) * native_h / 2}
    source = {name: float(value * (source_w / native_w if name in {"fx", "cx"}
                                  else source_h / native_h)) for name, value in native.items()}
    result = {"fx": source["fx"] * content_w / source_w,
              "fy": source["fy"] * content_h / source_h,
              "cx": source["cx"] * content_w / source_w + offset_x,
              "cy": source["cy"] * content_h / source_h + offset_y}
    if not all(math.isfinite(v) for v in result.values()) or min(result["fx"], result["fy"]) <= 0:
        raise ValueError("Camera projection produced invalid intrinsics")
    return result, transform


def projection_error_px(scene, camera, pose, transform):
    """Check exported xyzw/c2w/K against Blender's independent world projection."""
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Quaternion, Vector

    x, y, z, w = pose["quaternion"]
    reconstructed = Quaternion((w, x, y, z)).to_matrix().to_4x4()
    reconstructed.translation = Vector(pose["position"])
    inverse = reconstructed.inverted()
    K = pose["intrinsics"]
    content_w, content_h = transform["resized_size"]
    offset_x, offset_y = transform["offset_xy"]
    errors = []
    for local in [(0, 0, -2), (.3, .2, -2), (-.7, .4, -5), (.6, -.8, -5), (-1, -.5, -11)]:
        world = camera.matrix_world @ Vector(local)
        ndc = world_to_camera_view(scene, camera, world)
        expected = (ndc.x * content_w + offset_x, (1 - ndc.y) * content_h + offset_y)
        point = inverse @ world
        actual = (K["fx"] * point.x / -point.z + K["cx"],
                  K["fy"] * -point.y / -point.z + K["cy"])
        errors.append(max(abs(a - b) for a, b in zip(actual, expected)))
    error = max(errors)
    if not math.isfinite(error) or error > .01:
        raise ValueError(f"Exported camera disagrees with Blender projection by {error:.6f}px")
    return error


def export_case(case, out):
    import bpy

    if case not in CASES:
        raise ValueError(f"Unsupported camera case: {case}")
    out = Path(out)
    destination = out / "gt" / f"{case}.json"
    if destination.exists():
        raise FileExistsError(f"Camera GT already exists: {destination}")
    blend = paths.scene_blend(case)
    video_dir = paths.draft_dir(case, "P2_AA")
    reference = video_dir / f"{case}_P2_AA.mp4"
    preprocessing_path = video_dir / "preprocessing.json"
    preprocessing = json.loads(preprocessing_path.read_text())
    blend_hash, video_hash = sha256(blend), sha256(reference)
    if video_hash != preprocessing["draft_sha256"]:
        raise ValueError(f"{case}: reference video does not match preprocessing manifest")
    stream = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate", "-of", "json", str(reference)],
        text=True))["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    frame_count, fps = int(stream["nb_frames"]), float(preprocessing["fps"])
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    if (width, height, frame_count, fps) != (832, 480, 81, 16) or not math.isclose(numerator / denominator, fps):
        raise ValueError(f"{case}: expected the canonical 832x480, 81-frame, 16fps reference")
    indices = preprocessing["frame_indices"]
    if len(indices) != frame_count or preprocessing["frames"] != frame_count:
        raise ValueError(f"{case}: preprocessing frame count mismatch")
    if any(not isinstance(i, int) for i in indices) or any(b < a for a, b in zip(indices, indices[1:])):
        raise ValueError(f"{case}: invalid source-frame mapping")
    source_stream = preprocessing["source_metadata"]["streams"][0]
    source_size = [source_stream["width"], source_stream["height"]]
    bpy.ops.wm.open_mainfile(filepath=str(blend), use_scripts=False)
    scene = bpy.context.scene
    if scene.camera is None or scene.camera.data.type != "PERSP":
        raise ValueError(f"{case}: a perspective camera is required")
    native_size = [scene.render.resolution_x, scene.render.resolution_y]
    if native_size != [1280, 720] or source_size != [640, 360]:
        raise ValueError(f"{case}: native/source canvas changed; inspect the historical render mapping")
    trajectory, projection_errors, focal_lengths = [], [], []
    for frame, source_frame in enumerate(indices):
        blender_frame = source_to_blender_frame(source_frame, preprocessing["source_frames"],
                                                scene.frame_start, scene.frame_end)
        scene.frame_set(blender_frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        camera = scene.camera.evaluated_get(depsgraph)
        if max(abs(v - 1) for v in camera.matrix_world.to_scale()) > 1e-5:
            raise ValueError(f"{case}: scaled camera transform cannot be represented as a rigid c2w pose")
        projection = camera.calc_matrix_camera(depsgraph, x=native_size[0], y=native_size[1],
                                               scale_x=scene.render.pixel_aspect_x,
                                               scale_y=scene.render.pixel_aspect_y)
        intrinsics, transform = intrinsics_from_projection(projection, native_size, source_size, [width, height])
        rotation = camera.matrix_world.to_quaternion().normalized()
        if rotation.w < 0:
            rotation.negate()
        pose = {"frame": frame, "time_s": frame / fps, "source_frame": source_frame,
                "blender_frame": blender_frame,
                "position": list(camera.matrix_world.translation),
                "quaternion": [rotation.x, rotation.y, rotation.z, rotation.w],
                "intrinsics": intrinsics}
        projection_errors.append(projection_error_px(scene, camera, pose, transform))
        focal_lengths.append(float(camera.data.lens))
        trajectory.append(pose)
    if sha256(blend) != blend_hash:
        raise RuntimeError(f"{case}: source blend changed during export")
    result = {
        "schema": "worlddraft.blender_camera", "case_id": case, "source_cell": "P2_AA",
        "source_blend": repo_rel(blend), "source_blend_sha256": blend_hash,
        "reference_video": repo_rel(reference), "reference_video_sha256": video_hash,
        "preprocessing_path": repo_rel(preprocessing_path), "preprocessing_sha256": sha256(preprocessing_path),
        "source_video_sha256": preprocessing["source_sha256"], "source_size": source_size,
        "width": width, "height": height, "fps": fps, "frame_count": frame_count,
        "camera": scene.camera.name, "camera_type": "PERSP", "blender_version": bpy.app.version_string,
        "convention": "camera_to_world, OpenGL (x right, y up, looks -z), quaternion xyzw",
        "translation_units": "Blender scene units, not assumed meters",
        "pixel_convention": "top-left origin; principal point in pixel-boundary coordinates",
        "spatial_mapping": {"native_size": native_size, "native_to_source_scale": [.5, .5],
                            "source_to_input": transform},
        "temporal_mapping": {"input_to_source": "preprocessing.frame_indices",
                             "source_to_blender": "round(source_index * scene_frame_count / source_frames) + frame_start",
                             "scene_frame_range": [scene.frame_start, scene.frame_end],
                             "scene_fps": scene.render.fps / scene.render.fps_base,
                             "source_frame_count": preprocessing["source_frames"],
                             "renderer": "draft renderer: first output frame = Blender frame 1"},
        "evaluation_grid": {"fps": 8, "frame_count": 40, "first_time_s": 0, "last_time_s": 4.875},
        "camera_trajectory": trajectory, "dynamic_phrases": DYNAMIC_PHRASES[case],
        "dynamic_mask_notes": "Category masks target moving parts, retaining static buildings and support structures. "
                              "Boat masks can also remove moored boats; detector accuracy is not guaranteed.",
        "validation": {"projection_check": "5 world points per input frame against world_to_camera_view",
                       "max_projection_error_px": max(projection_errors),
                       "focal_length_mm_range": [min(focal_lengths), max(focal_lengths)],
                       "source_blend_unchanged": True},
        "role": "Blender-authored poses for diagnostics and intrinsics; scoring reference is ViPE(reference_video).",
        "code_sha256": sha256(Path(__file__)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"{case}: {len(trajectory)} poses; K={trajectory[0]['intrinsics']}; "
          f"projection max error={max(projection_errors):.6f}px; {destination}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Run directory; writes only its gt/ subdirectory")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    for case in dict.fromkeys(args.cases):
        export_case(case, args.out)


if __name__ == "__main__":
    main()
