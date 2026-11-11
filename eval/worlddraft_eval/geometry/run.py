"""Geometry stage: SAM 3 / SAM 2 tracking of the generated videos, entity matching and BBox IoU, identity features."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import time
import traceback

import numpy as np

from worlddraft_eval.camera.run import hash_file, hash_value, write_json
from worlddraft_eval.gt.entities import PROTOCOL, ROOT, load_protocol
from worlddraft_eval.common.paths import GT
from worlddraft_eval.common.io import CELLS, output_video, module_file
from worlddraft_eval.geometry import common as C, tracking as T, entity_metrics as E, memory_metrics as M, identity as I
from worlddraft_eval.common import model_config

MODELS = model_config.models()  # default: every model registered in eval/configs/models.yaml
TRACKERS = {"sam3": "facebook/sam3", "sam2": "facebook/sam2.1-hiera-large", "identity": "facebook/dinov2-base"}
METRICS = ["mean_iou", "mean_center_err", "miss_rate", "reappear_iou",
           "identity_revisit", "identity_short", "identity_cross"]


def environment():
    result = {}
    for name in ["transformers", "torch", "torchvision", "numpy", "scipy", "kernels"]:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def configure(protocol):
    settings = protocol["settings"]
    if settings["duration_s"] != C.EVAL_DURATION or settings["source_fps"] != 16:
        raise ValueError("This adapter requires a five-second, 16 fps Blender reference")
    # Dataset constants only: all upstream matching, thresholds and scoring stay unchanged.
    C.SOURCE_FPS = settings["source_fps"]
    T.LABEL_QUERY.update(protocol["prompts"])


def sample_for(job, gt, run_id):
    meta = C.video_meta(job["video"])
    w, h = gt["width"], gt["height"]
    from worlddraft_eval.common.spatial_mapping import input_crop, TRANSFORMS
    crop = input_crop(job["model"], [w, h], [meta["width"], meta["height"]])
    mapping = TRANSFORMS[job["model"]]
    extra = dict(spatial_mapping=mapping, source_fps=16, no_output_fitted_registration=True)
    sample = C.GeneratedSample(sample_id=f"{job['model']}/{job['case']}/{job['cell']}",
        episode=job["case"], camera="blender", representation=job["cell"], method=job["model"],
        run_id=run_id, video=job["video"], crop_xywh=crop, extra=extra, **meta)
    if max(C.eval_grid()) > (sample.frame_count - .5) / sample.fps + 1e-9:
        raise ValueError(f"Video does not cover the evaluation grid: {sample.video}")
    return sample


def code_fingerprint():
    paths = [Path(__file__), module_file("common.spatial_mapping"), PROTOCOL,
             *sorted(Path(T.__file__).parent.glob("*.py"))]
    return {str(p.relative_to(ROOT)): hash_file(p) for p in paths}


def prepare(out, gt_dir, cases=None, models=None, cells=None, workers=4):
    out = Path(out).resolve()
    if (out / "plan.json").exists():
        raise FileExistsError(out / "plan.json")
    protocol = load_protocol()
    configure(protocol)
    cases, models, cells = cases or list(protocol["cases"]), models or MODELS, cells or CELLS
    if not set(cases) <= set(protocol["cases"]) or not set(models) <= set(model_config.models()) or not set(cells) <= set(CELLS):
        raise ValueError("Unknown case, model or grid cell")
    if workers < 1:
        raise ValueError("workers must be positive")
    from worlddraft_eval.common.models import local
    model_paths = {kind: str(local(model_id).resolve()) for kind, model_id in TRACKERS.items()}
    model_hashes = {}
    for kind, directory in model_paths.items():
        files = sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix in {".json", ".safetensors", ".txt"})
        if not any(p.suffix == ".safetensors" for p in files):
            raise ValueError(f"Missing model weights: {directory}")
        model_hashes[kind] = {str(p): hash_file(p) for p in files}
    ground_truth, jobs, missing = {}, [], []
    for case in cases:
        path = Path(gt_dir).resolve() / f"{case}.json"
        gt = C.read_json(path)
        if gt["episode"] != case or gt["fps"] != 16 or len(gt["frames"]) != 81:
            raise ValueError(f"Invalid Blender GT: {path}")
        if gt["source"]["protocol_sha256"] != hash_file(PROTOCOL):
            raise ValueError(f"Stale entity definitions: {path}")
        frozen_gt = out / "inputs/gt" / f"{case}.json"
        write_json(frozen_gt, gt)
        ground_truth[case] = dict(path=str(frozen_gt), sha256=hash_file(frozen_gt))
        for model in models:
            for cell in cells:
                video = output_video(model, case, cell).resolve()
                job = dict(model=model, case=case, cell=cell, video=str(video))
                if not video.is_file():
                    missing.append(job)
                    continue
                sample = sample_for(job, gt, out.name)
                jobs.append(dict(job, video_sha256=hash_file(video), sample=sample.to_dict(),
                                 worker=len(jobs) % workers))
    config = dict(protocol=protocol, code=code_fingerprint(), model_paths=model_paths,
                  model_hashes=model_hashes, identity_entities="dynamic",
                  environment=environment(),
                  identity_protocol="upstream memory_identity; revisit only when a qualifying absence exists")
    plan = dict(created_at=time.time(), cases=cases, models=models, cells=cells, workers=workers,
                jobs=jobs, missing=missing, ground_truth=ground_truth, config=config, config_hash=hash_value(config))
    write_json(out / "inputs/entities.json", protocol)
    write_json(out / "plan.json", plan)
    print(f"Prepared {len(jobs)} videos; {len(missing)} missing -> {out}", flush=True)
    return plan


def artifact(out, kind, job):
    return Path(out) / kind / job["model"] / job["case"] / f"{job['cell']}.json"


def identity_score(video, memory, sample, loaded):
    if I._needed(memory, 1.0, memory["summary"]["grid_hz"])[1]:
        return I.evaluate(video, memory, sample, loaded)
    # Upstream's video iterator rejects an empty frame set; no crop pairs means N/A, not zero.
    return dict(schema="worlddraft.memory_identity", sample_id=sample.sample_id,
        status="not_applicable", reason="No predicted dynamic-entity crop pairs",
        summary=dict(identity_revisit=None, identity_short=None, identity_cross=None,
                     n_events=len(memory["events"]), n_events_scored=0, n_short_pairs=0, n_crops=0), events=[])


def process_job(out, plan, job, loaded, device):
    gt_info = plan["ground_truth"][job["case"]]
    if hash_file(job["video"]) != job["video_sha256"] or hash_file(gt_info["path"]) != gt_info["sha256"]:
        raise ValueError("Video or GT changed after preparation")
    key = hash_value(dict(config=plan["config_hash"], video=job["video_sha256"], gt=gt_info["sha256"], sample=job["sample"]))
    result_path, track_path = artifact(out, "metrics", job), artifact(out, "tracks", job)
    if result_path.exists():
        previous = C.read_json(result_path)
        if previous.get("status") == "complete" and previous.get("cache_key") == key:
            return previous
    sample, gt = C.GeneratedSample.from_dict(job["sample"]), C.read_json(gt_info["path"])
    settings = plan["config"]["protocol"]["settings"]
    paths = plan["config"]["model_paths"]
    prediction = C.read_json(track_path) if track_path.exists() else None
    if prediction is None or prediction.get("cache_key") != key:
        fallback = settings["sam3_fallback_engine_renders"] if job["model"] == model_config.reference() else settings["sam3_fallback"]
        prediction = T.run_sam3(job["video"], gt, sample, str(track_path), paths["sam3"], device,
            grid_hz=settings["grid_hz"], entities=settings["entities"], tracker=loaded["sam3"],
            score_thresh=settings["sam3_score_thresh"], fallback=fallback, sam2_model=paths["sam2"], sam2=loaded["sam2"])
        from transformers.models.sam3_video import modeling_sam3_video
        prediction["sam3_cv_postprocessing"] = modeling_sam3_video.cv_utils_kernel not in (None, False)
        prediction["cache_key"] = key
        write_json(track_path, prediction)
    ent = E.evaluate(gt, prediction, sample, grid_hz=settings["grid_hz"], entities=settings["entities"],
                     min_visible_area=settings["min_visible_area"], min_track_iou=settings["min_track_iou"])
    memory = M.evaluate(gt, prediction, sample, grid_hz=settings["grid_hz"], entities="dynamic")
    identity = identity_score(job["video"], memory, sample, loaded["identity"])
    for kind, value in [("entity", ent), ("memory", memory), ("identity", identity)]:
        write_json(artifact(out, kind, job), value)
    summary = {k: ent["summary"][k] for k in ["mean_iou", "mean_center_err", "miss_rate", "n_entities", "n_tracks", "n_assigned"]}
    summary.update({k: identity["summary"][k] for k in ["identity_revisit", "identity_short", "identity_cross", "n_events", "n_events_scored"]})
    summary.update(reappear_iou=memory["summary"]["reappear_iou"],
                   n_unassigned_tracks=len(ent["summary"]["unassigned_tracks"]),
                   fallback_entities=len(prediction["tracker"]["fallback_used"].get("entities", [])))
    result = dict(status="complete", cache_key=key, case=job["case"], model=job["model"], cell=job["cell"],
                  summary=summary, sample=job["sample"], tracks=str(track_path),
                  matching="Global track assignment for entity metrics; per-instant best match for memory/identity crops.")
    write_json(result_path, result)
    return result


def worker(out, index, device="cuda:0", limit=None):
    import torch

    out = Path(out)
    plan = C.read_json(out / "plan.json")
    if not 0 <= index < plan["workers"]:
        raise ValueError("Worker index outside plan")
    configure(plan["config"]["protocol"])
    if code_fingerprint() != plan["config"]["code"]:
        raise ValueError("Code or entity protocol changed after preparation")
    if environment() != plan["config"]["environment"]:
        raise ValueError("Model environment changed after preparation")
    with (out / f"worker-{index}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for files in plan["config"]["model_hashes"].values():
            if any(hash_file(path) != expected for path, expected in files.items()):
                raise ValueError("Model files changed after preparation")
        paths = plan["config"]["model_paths"]
        torch.manual_seed(0)
        torch.set_num_threads(4)
        loaded = dict(sam3=T._load_sam3(paths["sam3"], device), sam2=T._load_sam2(paths["sam2"], device),
                      identity=I._load(paths["identity"], device))
        jobs = [j for j in plan["jobs"] if j["worker"] == index]
        for n, job in enumerate(jobs[:limit] if limit else jobs, 1):
            started = time.time()
            try:
                result = process_job(out, plan, job, loaded, device)
                print(f"{n}/{len(jobs)} {job['model']} {job['case']} {job['cell']}: {result['summary']} ({time.time()-started:.1f}s)", flush=True)
            except Exception as exc:
                write_json(artifact(out, "metrics", job), dict(status="failed", case=job["case"], model=job["model"],
                    cell=job["cell"], error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()))
                traceback.print_exc()
                torch.cuda.empty_cache()
        write_json(out / f"worker-{index}.json", dict(finished_at=time.time(), attempted=min(limit or len(jobs), len(jobs))))


def report(out):
    out = Path(out)
    plan = C.read_json(out / "plan.json")
    rows, statuses = [], defaultdict(int)
    for job in plan["jobs"]:
        path = artifact(out, "metrics", job)
        result = C.read_json(path) if path.exists() else dict(status="pending")
        statuses[result["status"]] += 1
        rows.append(dict(model=job["model"], case=job["case"], cell=job["cell"],
                         status=result["status"], **result.get("summary", {})))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "video_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    aggregate = []
    for model in plan["models"]:
        for metric in METRICS:
            by_case = defaultdict(list)
            for row in rows:
                if row["model"] == model and row["status"] == "complete" and row.get(metric) is not None:
                    by_case[row["case"]].append(row[metric])
            aggregate.append(dict(model=model, metric=metric, n_videos=sum(map(len, by_case.values())), n_cases=len(by_case),
                                  value=float(np.mean([np.mean(v) for v in by_case.values()])) if by_case else None))
    with (out / "model_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "metric", "n_videos", "n_cases", "value"])
        writer.writeheader(); writer.writerows(aggregate)
    status = dict(expected=len(plan["jobs"]) + len(plan["missing"]), available=len(plan["jobs"]),
                  missing=len(plan["missing"]), **statuses)
    write_json(out / "status.json", status)
    print(json.dumps(status), flush=True)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "worker", "report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, default=GT / "entities")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--models", nargs="+", choices=model_config.models())
    parser.add_argument("--cells", nargs="+", choices=CELLS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out, args.gt_dir, args.cases, args.models, args.cells, args.workers)
    elif args.action == "worker":
        worker(args.out, args.index, args.device, args.limit)
    else:
        report(args.out)


if __name__ == "__main__":
    main()
