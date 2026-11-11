"""Camera protocol tests use synthetic trajectories and a fake estimator; no GPU."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from worlddraft_eval.camera import run as C


def trajectory(fps=16, factor=1.0, yaw=False):
    frames = []
    for i in range(5 * fps + 1):
        theta = np.pi / 2 * i / (5 * fps) if yaw else 0
        frames.append(dict(frame=i, position=[factor * i / fps, 0, 0],
                           quaternion=[0, np.sin(theta / 2), 0, np.cos(theta / 2)]))
    return dict(fps=fps, frames=frames)


def gt(case):
    return dict(case_id=case, width=832, height=480, dynamic_phrases=["person", "car"],
                camera_trajectory=[dict(intrinsics=dict(fx=400, fy=400, cx=416, cy=240))])


def test_exact_grid_and_native_canvas_intrinsics():
    assert C.P.eval_grid(8, 5) == [i / 8 for i in range(40)]
    assert C.fixed_intrinsics(gt("D01"), dict(width=1344, height=768)) == pytest.approx(
        [400 * 1344 / 832, 640, 672, 384])


def test_reference_identity_and_fps_mapping():
    result = C.score_trajectories(trajectory(16), trajectory(24))
    assert result["summary"] == pytest.approx(dict(RotErr_deg_mean=0, TransErr_rel_mean=0, CamMC_rel_mean=0), abs=1e-8)
    assert len(result["per_instant"]) == 40
    assert result["per_instant"][-1]["time_s"] == 4.875


def test_rot_units_and_original_translation_scale_invariance():
    rotation = C.score_trajectories(trajectory(factor=0), trajectory(factor=0, yaw=True))
    assert rotation["summary"]["RotErr_deg_mean"] == pytest.approx(43.875)
    assert rotation["summary"]["TransErr_rel_mean"] == 0
    # Translation is compared in shape only; no amplitude penalty is added.
    scaled = C.score_trajectories(trajectory(), trajectory(factor=.1))
    assert scaled["summary"]["TransErr_rel_mean"] < 1e-8
    stopped = C.score_trajectories(trajectory(), trajectory(factor=0))
    assert stopped["summary"]["TransErr_rel_mean"] == pytest.approx(.5, abs=1e-8)


def test_short_trajectory_is_rejected_instead_of_truncated():
    short = trajectory()
    short["frames"] = short["frames"][:40]
    with pytest.raises(ValueError, match="cover"):
        C.score_trajectories(trajectory(), short)


def test_plan_preserves_available_and_missing_and_reference_first(tmp_path, monkeypatch):
    out = tmp_path / "run"
    (out / "gt").mkdir(parents=True)
    def path(model, case, cell):
        return tmp_path / "videos" / model / case / f"{cell}.mp4"
    monkeypatch.setattr(C.Q, "output_video", path)
    for case in C.CASES:
        (out / "gt" / f"{case}.json").write_text(json.dumps(gt(case)))
        for model in C.MODELS:
            for cell in C.CELLS:
                if model == "MiniMax" and case != "D01":
                    continue
                target = path(model, case, cell)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"placeholder")
    plan = C.prepare(out)
    assert len(plan["jobs"]) == len(C.CASES)*len(C.MODELS)*len(C.CELLS)-63 and len(plan["missing"]) == 63
    assert all(j["status"] == "missing" for j in plan["missing"])
    for i, case in enumerate(C.CASES):
        jobs = [j for j in plan["jobs"] if j["case"] == case]
        assert {j["worker"] for j in jobs} == {i // 2}
        assert jobs[0]["reference"] and (jobs[0]["model"], jobs[0]["cell"]) == ("draft", "P2_AA")
    with pytest.raises(FileExistsError):
        C.prepare(out)


def test_estimator_is_reused_cache_is_resumable_and_changed_input_rejected(tmp_path, monkeypatch):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt("D01")))
    config = dict(seed=0, instants=C.P.eval_grid(8, 5))
    plan = dict(config=config, config_hash=C.hash_value(config),
                ground_truth={"D01":dict(path=str(gt_path), sha256=C.hash_file(gt_path))})
    class FakeEstimator:
        torch = SimpleNamespace(manual_seed=lambda seed: None)
        calls = 0
        def estimate(self, video, K, phrases, name):
            self.calls += 1
            assert K == [400, 400, 416, 240] and phrases == ["person", "car"]
            poses = np.tile(np.eye(4), (81, 1, 1))
            poses[:, 0, 3] = np.arange(81) / 16
            return poses, dict(fps=16, frames=81)
    engine = FakeEstimator()
    monkeypatch.setattr(C, "video_metadata", lambda path: dict(width=832, height=480, frame_count=81, fps=16))
    def job(model, cell, reference):
        video = tmp_path / f"{model}-{cell}.mp4"
        video.write_bytes(b"video")
        return dict(model=model, case="D01", cell=cell, reference=reference,
                    video=str(video), video_sha256=C.hash_file(video))
    ref = job("draft", "P2_AA", True)
    pred = job("MiniMax", "P1_AC", False)
    returned, ref_result = C.process_job(tmp_path, plan, ref, "/unused", engine)
    assert returned is engine and ref_result["reference"] and not any(ref_result["summary"].values())
    returned, result = C.process_job(tmp_path, plan, pred, "/unused", engine)
    assert returned is engine and result["status"] == "complete" and engine.calls == 2
    _, cached = C.process_job(tmp_path, plan, pred, "/unused", engine)
    assert cached == result and engine.calls == 2
    Path(pred["video"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        C.process_job(tmp_path, plan, pred, "/unused", engine)


def test_seedance_padding_intrinsics():
    from worlddraft_eval.common.spatial_mapping import input_crop
    assert input_crop('Seedance-2.5', (832, 480), (854, 480)) == [0, -6, 832, 492]
    k = C.fixed_intrinsics(gt('D01'), dict(width=854, height=480), 'Seedance-2.5')
    assert k == pytest.approx([400*854/832, 400*480/492, 427, 240])
    # Padding maps the original image edges inside the generated canvas.
    assert 6*480/492 > 0 and 486*480/492 < 480
