"""Camera stage: ViPE trajectories with fixed intrinsics, scored against each scene's draft P2_AA."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
import traceback

import numpy as np

from worlddraft_eval.common import io as Q
from worlddraft_eval.camera import vipe as P
from worlddraft_eval.common import model_config

CASES = [f"D{i:02d}" for i in range(1, 9)]
MODELS = model_config.models()  # default: every model registered in eval/configs/models.yaml
CELLS = list(Q.CELLS)
VIPE_COMMIT = "95a8816947602ddc26fcb7a80bea4f9313059578"


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(out, vipe_root=None, models=None):
    models = MODELS if models is None else models
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise FileExistsError("plan.json already exists; use --worker to resume this run")
    config = dict(protocol="cami2v_all_motion",
                  vipe_commit=VIPE_COMMIT, vendor_sha256=hash_file(P.__file__),
                  runner_sha256=hash_file(__file__), pipeline="pose_only",
                  pipeline_overrides=P.VIPE_PIPELINES["pose_only"], intrinsics="fixed_gt_first_frame",
                  intrinsics_resize="known_seedance_padding_then_whole_canvas; other_models_whole_condition_canvas", seed=0,
                  reference_model="draft", reference_cell="P2_AA", grid_hz=8.0, duration_s=5.0,
                  instants=P.eval_grid(8.0, 5.0))
    jobs, missing, ground_truth = [], [], {}
    for index, case in enumerate(CASES):
        gt_path = out / "gt" / f"{case}.json"
        gt = json.loads(gt_path.read_text())
        if gt.get("case_id") != case or not gt.get("camera_trajectory"):
            raise ValueError(f"Invalid camera GT: {gt_path}")
        if not gt.get("width") or not gt.get("height") or "dynamic_phrases" not in gt:
            raise ValueError(f"GT lacks condition size or dynamic_phrases: {gt_path}")
        ground_truth[case] = dict(path=str(gt_path), sha256=hash_file(gt_path))
        pairs = [("draft", "P2_AA")] + [(m, c) for m in models for c in CELLS if (m, c) != ("draft", "P2_AA")]
        for model, cell in pairs:
            video = Q.output_video(model, case, cell).resolve()
            job = dict(model=model, case=case, cell=cell, video=str(video), worker=index // 2,
                       reference=(model, cell) == ("draft", "P2_AA"))
            if video.is_file():
                jobs.append(dict(job, video_sha256=hash_file(video), status="pending"))
            else:
                missing.append(dict(job, status="missing"))
    plan = dict(created_at=time.time(), workers=4, cases=CASES, models=models, cells=CELLS,
                config=config, config_hash=hash_value(config), ground_truth=ground_truth,
                vipe_root=str(Path(vipe_root).resolve()) if vipe_root else None,
                jobs=jobs, missing=missing, expected_videos=len(CASES) * len(models) * len(CELLS))
    write_json(out / "plan.json", plan)
    print(f"Prepared {len(jobs)} camera videos; {len(missing)} missing; 4 workers grouped by case", flush=True)
    return plan


def video_metadata(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    try:
        result = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                      frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), fps=float(cap.get(cv2.CAP_PROP_FPS)))
        if not cap.isOpened() or not all(np.isfinite(v) and v > 0 for v in result.values()):
            raise ValueError(f"Cannot read video metadata: {path}")
        return result
    finally:
        cap.release()


def fixed_intrinsics(gt, metadata, model=None):
    k = gt["camera_trajectory"][0]["intrinsics"]
    sx, sy = metadata["width"] / gt["width"], metadata["height"] / gt["height"]
    cy = k["cy"]
    config = model_config.get(model) if model is not None else {}
    if config.get('transform') == 'pad':
        assert [gt['width'], gt['height']] == list(config['input_size'])
        sy = metadata['height'] / (gt['height'] + config['top'] + config['bottom'])
        cy += config['top']
    result = [float(k["fx"] * sx), float(k["fy"] * sy), float(k["cx"] * sx), float(cy * sy)]
    if not np.isfinite(result).all() or min(result[:2]) <= 0:
        raise ValueError("Invalid fixed intrinsics")
    return result


def score_trajectories(reference, prediction, instants=None):
    instants = P.eval_grid(8.0, 5.0) if instants is None else instants
    pg, qg = P._sample_traj(reference, instants, fps=reference["fps"])
    pp, qp = P._sample_traj(prediction, instants, fps=prediction["fps"])
    if not all(np.isfinite(a).all() for a in (pg, qg, pp, qp)):
        raise ValueError("Non-finite camera trajectory")
    raw, per_instant = P.cami2v_metrics(pg, qg, pp, qp, return_per_instant=True)
    summary = dict(RotErr_deg_mean=float(np.degrees(raw["RotErr_mean"])),
                   TransErr_rel_mean=raw["TransErr_rel_mean"], CamMC_rel_mean=raw["CamMC_rel_mean"])
    if not all(np.isfinite(v) for v in summary.values()):
        raise ValueError("Non-finite camera metric")
    return dict(summary=summary, official_summary=raw,
                per_instant=[dict(time_s=t, **r) for t, r in zip(instants, per_instant)])


def artifact_path(out, kind, job):
    return Path(out) / kind / job["model"] / job["case"] / f"{job['cell']}.json"


def process_job(out, plan, job, vipe_root, estimator=None):
    """Return the reusable estimator and metric; cache only matching inputs/configuration."""
    gt_info = plan["ground_truth"][job["case"]]
    if hash_file(job["video"]) != job["video_sha256"] or hash_file(gt_info["path"]) != gt_info["sha256"]:
        raise ValueError("Video or camera GT changed after preparation")
    gt = json.loads(Path(gt_info["path"]).read_text())
    pose_path = artifact_path(out, "poses", job)
    pose_key = hash_value(dict(video=job["video_sha256"], gt=gt_info["sha256"], config=plan["config_hash"]))
    reference_job = dict(model="draft", case=job["case"], cell="P2_AA")
    reference_path = artifact_path(out, "poses", reference_job)
    if not job["reference"]:
        reference_metric = artifact_path(out, "metrics", reference_job)
        if not reference_metric.exists() or json.loads(reference_metric.read_text()).get("status") != "complete":
            raise ValueError("Reference draft/P2_AA estimation failed or is unavailable")
    if pose_path.exists():
        trajectory = json.loads(pose_path.read_text())
        if trajectory.get("cache_key") != pose_key:
            raise ValueError("Cached trajectory input/config hash mismatch; use a new run directory")
    else:
        metadata = video_metadata(job["video"])
        K = fixed_intrinsics(gt, metadata, job["model"])
        if estimator is None:
            estimator = P.VipeEstimator(str(vipe_root), device="cuda:0", vipe_pipeline="pose_only")
        random.seed(plan["config"]["seed"])
        np.random.seed(plan["config"]["seed"])
        estimator.torch.manual_seed(plan["config"]["seed"])
        c2w, info = estimator.estimate(job["video"], K, gt["dynamic_phrases"],
                                      name=f"{job['model']}__{job['case']}__{job['cell']}")
        if len(c2w) != metadata["frame_count"] or not np.isfinite(c2w).all():
            raise ValueError("Estimated camera trajectory has wrong frame count or non-finite poses")
        if not np.isclose(float(info["fps"]), metadata["fps"], rtol=1e-5):
            raise ValueError("ViPE FPS disagrees with input video")
        trajectory = P.c2w_cv_to_traj(c2w, metadata["fps"], info)
        trajectory.update(model=job["model"], case=job["case"], cell=job["cell"], reference=job["reference"],
                          cache_key=pose_key, config_hash=plan["config_hash"], input_sha256=job["video_sha256"],
                          gt_sha256=gt_info["sha256"], video=job["video"], video_metadata=metadata)
        write_json(pose_path, trajectory)
    reference = trajectory if job["reference"] else json.loads(reference_path.read_text())
    if reference.get("config_hash") != plan["config_hash"]:
        raise ValueError("Reference trajectory uses another configuration")
    metric_key = hash_value(dict(pose=hash_file(pose_path), reference=hash_file(reference_path), config=plan["config_hash"]))
    metric_path = artifact_path(out, "metrics", job)
    if metric_path.exists():
        cached = json.loads(metric_path.read_text())
        if cached.get("status") == "complete" and cached.get("cache_key") == metric_key:
            return estimator, cached
    result = dict(status="complete", model=job["model"], case=job["case"], cell=job["cell"],
                  reference=job["reference"], cache_key=metric_key, config_hash=plan["config_hash"],
                  video=job["video"], input_sha256=job["video_sha256"], trajectory_path=str(pose_path),
                  reference_trajectory_path=str(reference_path), reference_model="draft", reference_cell="P2_AA",
                  seconds=trajectory.get("estimator", {}).get("seconds"),
                  normalization="relative_to_first; own_max_translation_norm_plus_1e-9", grid_hz=8.0,
                  n_instants=len(plan["config"]["instants"]),
                  **score_trajectories(reference, trajectory, plan["config"]["instants"]))
    if job["reference"]:
        # Identity comparison is a declared reference baseline, not estimator validation.
        result["summary"] = {k: 0.0 for k in result["summary"]}
    write_json(metric_path, result)
    return estimator, result


def worker(out, index, vipe_root=None):
    out = Path(out).resolve()
    plan = json.loads((out / "plan.json").read_text())
    if index not in range(plan["workers"]):
        raise ValueError("Invalid worker index")
    root = Path(vipe_root or plan.get("vipe_root") or "")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (revision != plan["config"]["vipe_commit"] or hash_file(P.__file__) != plan["config"]["vendor_sha256"]
            or hash_file(__file__) != plan["config"]["runner_sha256"]):
        raise ValueError("ViPE commit or camera runner/vendor code differs from prepared configuration")
    with (out / f"worker-{index}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(worker=index, status="running", started_at=time.time(), current_job=None,
                     complete=0, error=0, jobs=[])
        estimator = None
        for job in (j for j in plan["jobs"] if j["worker"] == index):
            started = time.time()
            state["current_job"] = f"{job['model']}/{job['case']}/{job['cell']}"
            write_json(out / f"worker-{index}.json", state)
            print(f"Starting {state['current_job']}", flush=True)
            try:
                estimator, result = process_job(out, plan, job, root, estimator)
            except Exception as exc:
                traceback.print_exc()
                result = dict(status="error", model=job["model"], case=job["case"], cell=job["cell"],
                              reference=job["reference"], error=f"{type(exc).__name__}: {exc}",
                              config_hash=plan["config_hash"], input_sha256=job["video_sha256"],
                              video=job["video"], summary=None)
                write_json(artifact_path(out, "metrics", job), result)
            elapsed = time.time() - started
            state[result["status"]] += 1
            state["jobs"].append(dict(model=job["model"], case=job["case"], cell=job["cell"],
                                      status=result["status"], elapsed_seconds=elapsed))
            state["elapsed_seconds"] = time.time() - state["started_at"]
            state["current_job"] = None
            write_json(out / f"worker-{index}.json", state)
            print(f"{job['model']}/{job['case']}/{job['cell']}: {result['status']} ({elapsed:.1f}s)", flush=True)
        state.update(status="complete_with_errors" if state["error"] else "complete", finished_at=time.time())
        write_json(out / f"worker-{index}.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vipe-root", type=Path)
    parser.add_argument("--models", nargs="+", choices=model_config.models())
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--worker", type=int)
    actions.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.out, args.vipe_root, args.models)
    elif args.worker is not None:
        worker(args.out, args.worker, args.vipe_root)
    else:
        from worlddraft_eval.camera.report import report
        report(args.out)


if __name__ == "__main__":
    main()
