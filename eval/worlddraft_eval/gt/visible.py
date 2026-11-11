"""Export canonical, occlusion-aware contract GT from the final P2_AA blends.

Run with the separate bpy environment; no scene file is ever saved. Coordinates
are measured in the original MP4 canvas, including its encoder padding. The
input-video letterbox is a separate, explicitly recorded transform.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from worlddraft_eval.gt.contracts import DEFAULT_CONTRACTS_DIR, load_contract
from worlddraft_eval.common import paths


DEFAULT_INPUT_INDICES = (0, 11, 23, 34, 46, 57, 69, 80)
GEOMETRY_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "VOLUME"}


class NoUsableEntities(ValueError):
    """The selected scene geometry cannot provide per-entity bounding boxes."""


def repo_rel(path) -> str:
    """Record paths as data/... or repo-relative so GT files carry no machine paths."""
    return paths.record(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_to_blender_frame(source_index: int, source_frames: int,
                            frame_start: int, frame_end: int) -> int:
    """Reproduce render_palette_contrast.py's round(k * N / 50) + start."""
    if not 0 <= source_index < source_frames:
        raise ValueError(f"Source index {source_index} outside {source_frames} frames")
    return round(source_index * (frame_end - frame_start + 1) / source_frames) + frame_start


def letterbox_transform(source_size, target_size) -> dict:
    width, height = source_size
    target_width, target_height = target_size
    scale = min(target_width / width, target_height / height)
    resized_width, resized_height = round(width * scale), round(height * scale)
    return {
        "source_size": [width, height], "target_size": [target_width, target_height],
        "resized_size": [resized_width, resized_height],
        "offset_xy": [(target_width - resized_width) // 2, (target_height - resized_height) // 2],
        "rule": "scale=min(target_w/source_w,target_h/source_h); round dimensions; floor center",
        "rule_evidence": "measured on the published draft videos",
    }


def preprocessing_transform(preprocessing):
    """Compose recorded resize/pad stages instead of inventing a new letterbox."""
    source = [int(preprocessing['source_metadata']['streams'][0][k]) for k in ('width','height')]
    target = list(map(int, preprocessing['size'].split('*')))
    chain = preprocessing.get('spatial_transform_chain')
    if not chain:
        return letterbox_transform(source, target)
    current = source; scale = [1., 1.]; offset = [0., 0.]
    for stage in chain:
        if list(stage['source_size']) != current:
            raise ValueError('Discontinuous preprocessing transform')
        for axis in (0, 1):
            ratio = stage['resized_size'][axis] / current[axis]
            if ratio <= 0:raise ValueError('Invalid preprocessing scale')
            offset[axis] = offset[axis]*ratio + stage['offset_xy'][axis]
            scale[axis] *= ratio
        current = list(stage['target_size'])
    if current != target:raise ValueError('Preprocessing canvas mismatch')
    affine = [[scale[0],0,offset[0]],[0,scale[1],offset[1]]]
    if 'source_to_condition_affine' in preprocessing and not np.allclose(affine,preprocessing['source_to_condition_affine'],atol=1e-9):
        raise ValueError('Recorded affine disagrees with transform chain')
    return dict(source_size=source,target_size=target,resized_size=[source[i]*scale[i] for i in (0,1)],
                offset_xy=offset,stages=chain,rule='composed recorded resize and pad stages')


def source_point_to_input(point, transform):
    if point is None:
        return None
    resized_w, resized_h = transform["resized_size"]
    offset_x, offset_y = transform["offset_xy"]
    target_w, target_h = transform["target_size"]
    return [(point[0] * resized_w + offset_x) / target_w,
            (point[1] * resized_h + offset_y) / target_h]


def collect_entities(scene, case_id: str, contract=None, contracts_dir=None):
    """Group actual objects under selected roots; independent parts also work.

    A pattern may match a root name or its stable scene identifier. When both
    an ancestor and a descendant match, only the ancestor is an entity, so no
    object is assigned to two masks. Distinct objects sharing mesh *data* remain
    distinct physical instances.
    """
    contract = contract if contract is not None else load_contract(case_id, contracts_dir)
    if contract.get("skip_reason"):
        raise NoUsableEntities(contract["skip_reason"])
    pattern = re.compile(contract["gt_root_pattern"])
    candidates = [obj for obj in scene.objects if any(pattern.fullmatch(str(value))
                  for value in (obj.name, obj.get("entity_id", ""), obj.get("vehicle_id", "")))]
    candidate_names = {obj.name for obj in candidates}

    def has_selected_ancestor(obj):
        parent = obj.parent
        while parent is not None:
            if parent.name in candidate_names:
                return True
            parent = parent.parent
        return False

    roots = [obj for obj in candidates if not has_selected_ancestor(obj)]
    references = {name for attribute in contract["attributes"] for name in attribute["entities"]}

    def entity_id(obj):
        identifiers = (obj.name, str(obj.get("entity_id", "")), str(obj.get("vehicle_id", "")))
        return next((value for value in identifiers if value in references),
                    str(obj.get("entity_id", obj.get("vehicle_id", obj.name))))

    roots.sort(key=entity_id)
    entities = []
    claimed = set()
    for root in roots:
        members = [obj for obj in [root, *root.children_recursive] if obj.type in GEOMETRY_TYPES]
        if not members:
            continue
        names = {obj.name for obj in members}
        if claimed.intersection(names):
            raise ValueError(f"Semantic entities overlap at {root.name}")
        claimed.update(names)
        entities.append({"instance_id": len(entities) + 1,
                         "entity_id": entity_id(root),
                         "root": root, "objects": members})
    if not entities:
        raise NoUsableEntities(f"{case_id}: pattern {contract['gt_root_pattern']!r} selects no renderable entities")
    identifiers = [entity["entity_id"] for entity in entities]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{case_id}: selected roots have duplicate entity identifiers")
    missing = references.difference(identifiers)
    if missing:
        raise NoUsableEntities(f"{case_id}: contracted entities absent from selected renderable roots: {sorted(missing)}")
    return entities


def id_palette(count: int) -> np.ndarray:
    """VOC bit-interleaved palette: well-separated, deterministic instance colors."""
    palette = np.zeros((count + 1, 3), dtype=np.uint8)
    for index in range(1, count + 1):
        value = index
        for bit in range(8):
            for channel in range(3):
                palette[index, channel] |= ((value >> channel) & 1) << (7 - bit)
            value >>= 3
    return palette


def configure_id_render(scene, entities, palette, render_size, samples, threads):
    import bpy

    def material(name, rgb):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (*[float(v) / 255 for v in rgb], 1)
        emission.inputs["Strength"].default_value = 1
        mat.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
        return mat

    materials = [material(f"evaluation_instance_{i:03d}", color) for i, color in enumerate(palette)]
    object_ids = {obj.name: entity["instance_id"] for entity in entities for obj in entity["objects"]}
    for obj in scene.objects:
        if obj.type not in GEOMETRY_TYPES or not hasattr(obj.data, "materials"):
            continue
        mat = materials[object_ids.get(obj.name, 0)]
        # Object-linked slots preserve shared mesh geometry and polygon indices.
        if not obj.material_slots:
            obj.data.materials.append(materials[0])
        for slot in obj.material_slots:
            slot.link = "OBJECT"
            slot.material = mat
    for layer in scene.view_layers:
        layer.material_override = None
    world = bpy.data.worlds.new("evaluation_black_world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0, 0, 0, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0
    scene.world = world
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.render.threads_mode = "FIXED"
    scene.render.threads = threads
    scene.render.resolution_x, scene.render.resolution_y = render_size
    scene.render.resolution_percentage = 100
    scene.render.use_motion_blur = False
    scene.render.film_transparent = False
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Raw"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    scene.view_settings.use_curve_mapping = False
    scene.render.dither_intensity = 0
    # Tiny reconstruction filter prevents identity colors blending at boundaries.
    scene.cycles.pixel_filter_type = "BOX"
    scene.cycles.filter_width = 0.01
    if scene.camera and hasattr(scene.camera.data, "dof"):
        scene.camera.data.dof.use_dof = False


def decode_instance_image(image_path: Path, palette: np.ndarray, source_size):
    from PIL import Image
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.int16)
    # Exact palette matches are expected with Raw color management; nearest color
    # is used for subpixel edge samples and their count is explicitly reported.
    best_dist = np.full(rgb.shape[:2], np.inf)
    ids = np.zeros(rgb.shape[:2], dtype=np.uint16)
    for index, color in enumerate(palette):
        distance = ((rgb.astype(np.float32) - color.astype(np.float32)) ** 2).sum(axis=2)
        selected = distance < best_dist
        ids[selected] = index
        best_dist[selected] = distance[selected]
    width, height = source_size
    padded = np.zeros((height, width), dtype=np.uint16)
    padded[:ids.shape[0], :ids.shape[1]] = ids
    return padded, {"non_palette_pixel_count": int((best_dist > 3).sum()),
                    "max_palette_distance_rgb": float(np.sqrt(best_dist.max()))}


def video_frame(path: Path, index: int):
    from PIL import Image
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{index})",
               "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"]
    return Image.open(io.BytesIO(subprocess.check_output(command))).convert("RGB")


def write_overlay(condition_path, input_index, labels, palette, transform, rows, output_path):
    from PIL import Image, ImageDraw
    canvas = video_frame(condition_path, input_index)
    target_w, target_h = transform["target_size"]
    if canvas.size != (target_w, target_h):
        raise ValueError(f"Condition dimensions {canvas.size} != {(target_w, target_h)}")
    target_labels = labels
    for stage in transform.get('stages', [transform]):
        resized = Image.fromarray(target_labels).resize(tuple(stage['resized_size']), Image.Resampling.NEAREST)
        tw, th = stage['target_size']; x, y = stage['offset_xy']
        target_labels = np.zeros((th, tw), dtype=np.uint16)
        target_labels[y:y+resized.height, x:x+resized.width] = np.asarray(resized)
    pixels = np.array(canvas)
    selected = target_labels > 0
    pixels[selected] = (0.6 * pixels[selected] + 0.4 * palette[target_labels[selected]]).astype(np.uint8)
    canvas = Image.fromarray(pixels)
    draw = ImageDraw.Draw(canvas)
    for row in rows:
        if row["visible"]:
            cx, cy = row["input_center_xy"]
            px, py = cx * target_w, cy * target_h
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline="white", width=1)
            draw.text((px + 3, py + 3), str(row["instance_id"]), fill="white", stroke_width=1, stroke_fill="black")
    draw.text((6, 6), f"P2_AA input frame {input_index}; rendered GT mask overlay", fill="white", stroke_width=1, stroke_fill="black")
    canvas.save(output_path)


def write_skipped_case(case_id, output_dir, reason, contract, contracts_dir=None):
    """Persist an explicit non-scoreable result, rather than an empty score."""
    destination = Path(output_dir).resolve() / case_id
    destination.mkdir(parents=True, exist_ok=True)
    contract_path = Path(contracts_dir or DEFAULT_CONTRACTS_DIR).resolve() / f"{case_id}.json"
    manifest = {"schema_version": "entity-spatial-gt", "case_id": case_id,
                "status": "skipped", "skip_reason": str(reason),
                "contract": contract, "contract_path": repo_rel(contract_path),
                "contract_sha256": sha256(contract_path),
                "entity_count": 0, "entities": [], "frames": []}
    (destination / "gt.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"case": case_id, "status": "skipped", "skip_reason": str(reason)}), flush=True)
    return manifest


def export_case(case_id: str, root: Path, output_dir: Path,
                input_indices=DEFAULT_INPUT_INDICES, samples=1, threads=8,
                min_visible_pixels=1, overlays=True, contracts_dir=None):
    contract = load_contract(case_id, contracts_dir)
    if contract.get("skip_reason"):
        return write_skipped_case(case_id, output_dir, contract["skip_reason"], contract, contracts_dir)

    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from PIL import Image

    root = root.resolve()
    blend_path = root / "scenes" / case_id / f"{case_id}_P2_AA.blend"
    preprocessing_path = root / "drafts" / case_id / f"{case_id}_P2_AA" / "preprocessing.json"
    preprocessing = json.loads(preprocessing_path.read_text())
    stream = preprocessing["source_metadata"]["streams"][0]
    source_size = (int(stream["width"]), int(stream["height"]))
    target_size = tuple(int(value) for value in preprocessing["size"].split("*"))
    transform = preprocessing_transform(preprocessing)
    condition_path = preprocessing_path.parent / f"{case_id}_P2_AA.mp4"
    condition_hash = sha256(condition_path)
    if condition_hash != preprocessing["draft_sha256"]:
        raise ValueError(f"Condition video hash mismatch: {condition_path}")
    input_indices = sorted(set(input_indices))
    frame_indices = preprocessing["frame_indices"]
    if not input_indices or any(index < 0 or index >= len(frame_indices) for index in input_indices):
        raise ValueError("Input index out of preprocessing mapping range")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    if scene.camera is None:
        raise ValueError(f"No active camera in {blend_path}")
    original_size = [scene.render.resolution_x, scene.render.resolution_y]
    render_size = [int(value * 0.5) for value in original_size]
    padded_size = [value + value % 2 for value in render_size]
    if padded_size != list(source_size):
        raise ValueError(f"Historical half-size render + even padding {padded_size} != source {source_size}")
    try:
        entities = collect_entities(scene, case_id, contract)
    except NoUsableEntities as exc:
        return write_skipped_case(case_id, output_dir, str(exc), contract, contracts_dir)
    palette = id_palette(len(entities))
    destination = output_dir.resolve() / case_id
    for name in ("rendered_ids", "instances", "overlays"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "entity-spatial-gt", "case_id": case_id, "status": "complete",
        "contract": contract,
        "contract_path": repo_rel(Path(contracts_dir or DEFAULT_CONTRACTS_DIR) / f"{case_id}.json"),
        "contract_sha256": sha256(Path(contracts_dir or DEFAULT_CONTRACTS_DIR) / f"{case_id}.json"),
        "entity_selection": "contract pattern fullmatch on object name, entity_id or vehicle_id; selected ancestors own all renderable descendants",
        "source_cell": f"{case_id}_P2_AA", "source_blend": repo_rel(blend_path),
        "source_blend_sha256": sha256(blend_path), "blender_version": bpy.app.version_string,
        "preprocessing_path": repo_rel(preprocessing_path), "preprocessing_sha256": sha256(preprocessing_path),
        "source_video_sha256": preprocessing["source_sha256"],
        "condition_path": repo_rel(condition_path), "condition_sha256": condition_hash,
        "source_size": list(source_size), "source_frame_count": preprocessing["source_frames"],
        "scene_frame_range": [scene.frame_start, scene.frame_end],
        "scene_fps": scene.render.fps / scene.render.fps_base,
        "camera": scene.camera.name, "camera_type": scene.camera.data.type,
        "entity_count": len(entities), "min_visible_pixels": min_visible_pixels,
        "coordinate_convention": "top-left origin; normalized source-video canvas; pixel-center mask centroid; exclusive bbox max",
        "position_measurement": "visible instance-mask centroid (not root origin or bbox midpoint)",
        "rendering": {"engine": "CYCLES", "device": "CPU", "samples": samples,
                      "threads": threads, "native_scene_size": original_size, "render_size": render_size,
                      "padding_right_bottom": [source_size[i] - render_size[i] for i in range(2)],
                      "method": "unique flat emission per semantic entity; other geometry opaque black; Raw color transform",
                      "geometry_camera_preserved": True, "motion_blur": False, "depth_of_field": False},
        "temporal_mapping": {
            "model_input_indices": input_indices, "model_input_fps": preprocessing["fps"],
            "input_to_source": "exact preprocessing.frame_indices",
            "source_to_blender": "round(source_index * scene_frame_count / source_frame_count) + frame_start",
            "source_to_blender_evidence": "draft renderer: first output frame = Blender frame 1",
            "source_mapping_status": "reconstructed_from_historical_renderer; verify overlays before formal scoring",
        },
        "letterbox": transform,
        "limitations": [
            "Visibility is geometric at source resolution, not a perceptual minimum-size judgment.",
            "All surfaces are treated as opaque for ID visibility; original transparency is not simulated.",
            "Depth of field and motion blur are disabled for geometric labels.",
            "Historical frame mapping requires visual corroboration against the saved input MP4.",
        ],
        "entities": [{"instance_id": e["instance_id"], "entity_id": e["entity_id"],
                      "root_object": e["root"].name, "member_objects": [obj.name for obj in e["objects"]],
                      "id_color_rgb": palette[e["instance_id"]].tolist()} for e in entities],
        "frames": [],
    }
    configure_id_render(scene, entities, palette, render_size, samples, threads)
    source_fps_num, source_fps_den = stream["r_frame_rate"].split("/")
    source_fps = float(source_fps_num) / float(source_fps_den)
    current_frame = scene.frame_start - 1
    for input_index in input_indices:
        source_index = int(frame_indices[input_index])
        blender_frame = source_to_blender_frame(source_index, preprocessing["source_frames"], scene.frame_start, scene.frame_end)
        for frame in range(current_frame + 1, blender_frame + 1):
            scene.frame_set(frame)
        current_frame = blender_frame
        scene.frame_set(blender_frame)
        tag = f"source_{source_index:03d}_blend_{blender_frame:03d}"
        rendered_path = destination / "rendered_ids" / f"{tag}.png"
        scene.render.filepath = str(rendered_path)
        bpy.ops.render.render(write_still=True)
        labels, color_diagnostics = decode_instance_image(rendered_path, palette, source_size)
        instance_path = destination / "instances" / f"{tag}.png"
        Image.fromarray(labels).save(instance_path)
        rows = []
        for entity in entities:
            ys, xs = np.where(labels == entity["instance_id"])
            area = int(xs.size)
            center = [(float(xs.mean()) + 0.5) / source_size[0], (float(ys.mean()) + 0.5) / source_size[1]] if area else None
            bbox = [float(xs.min()) / source_size[0], float(ys.min()) / source_size[1],
                    float(xs.max() + 1) / source_size[0], float(ys.max() + 1) / source_size[1]] if area else None
            point = world_to_camera_view(scene, scene.camera, entity["root"].matrix_world.translation)
            root_xy = [float(point.x) * render_size[0] / source_size[0],
                       (1 - float(point.y)) * render_size[1] / source_size[1]]
            rows.append({"instance_id": entity["instance_id"], "entity_id": entity["entity_id"],
                         "visible": area >= min_visible_pixels, "mask_area": area,
                         "center_xy": center, "bbox_xyxy": bbox,
                         "projected_root_xy": root_xy, "root_camera_depth": float(point.z),
                         "input_center_xy": source_point_to_input(center, transform)})
        overlay_path = destination / "overlays" / f"{tag}.png"
        if overlays:
            write_overlay(condition_path, input_index, labels, palette, transform, rows, overlay_path)
        manifest["frames"].append({
            "model_input_index": input_index, "source_index": source_index,
            "blender_frame": blender_frame, "time": source_index / source_fps,
            "source_time_s": source_index / source_fps,
            "model_input_time_s": input_index / preprocessing["fps"],
            "blender_time_s": (blender_frame - scene.frame_start) / manifest["scene_fps"],
            "visible_count": sum(row["visible"] for row in rows), "entities": rows,
            "instance_mask_path": repo_rel(instance_path), "rendered_id_path": repo_rel(rendered_path),
            "overlay_path": repo_rel(overlay_path) if overlays else None,
            "color_diagnostics": color_diagnostics,
        })
        print(json.dumps({"case": case_id, "source_index": source_index, "blender_frame": blender_frame,
                          "visible_count": manifest["frames"][-1]["visible_count"], "overlay": str(overlay_path)}), flush=True)
    manifest["code_sha256"] = sha256(Path(__file__))
    manifest["source_blend_unchanged"] = sha256(blend_path) == manifest["source_blend_sha256"]
    if not manifest["source_blend_unchanged"]:
        raise RuntimeError(f"Source blend changed during export: {blend_path}")
    if not any(frame["visible_count"] for frame in manifest["frames"]):
        manifest["status"] = "skipped"
        manifest["skip_reason"] = "Selected entities have no visible bounding boxes at any sampled time"
    (destination / "gt.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", required=True,
                        help="Case identifiers (A01 through F08); each requires a contract")
    parser.add_argument("--contracts-dir", type=Path, default=DEFAULT_CONTRACTS_DIR)
    parser.add_argument("--root", type=Path, default=paths.DATA, help="Data directory (drafts/, scenes/)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-indices", type=int, nargs="+", default=DEFAULT_INPUT_INDICES)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--min-visible-pixels", type=int, default=1)
    parser.add_argument("--no-overlays", action="store_true")
    args = parser.parse_args()
    if min(args.samples, args.threads, args.min_visible_pixels) < 1:
        parser.error("samples, threads and min-visible-pixels must be positive")
    for case_id in args.cases:
        export_case(case_id, args.root, args.output_dir, args.input_indices,
                    args.samples, args.threads, args.min_visible_pixels, not args.no_overlays,
                    args.contracts_dir)


if __name__ == "__main__":
    main()
