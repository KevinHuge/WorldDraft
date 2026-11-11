import json
from pathlib import Path

import numpy as np
import pytest

from worlddraft_eval.gt import entities as G
from worlddraft_eval.geometry import run as R
from worlddraft_eval.common.paths import GT
from worlddraft_eval.geometry import common as C, entity_metrics as E, memory_metrics as M


@pytest.fixture
def scene(monkeypatch):
    monkeypatch.setattr(C, "SOURCE_FPS", 16)
    gt = dict(entities={"a": dict(label="ball", dynamic=True), "b": dict(label="ball", dynamic=True)},
              frames=[dict(boxes={"a": dict(visible=True, box=[10, 10, 30, 30]),
                                 "b": dict(visible=True, box=[70, 10, 90, 30])}) for _ in range(81)])
    sample = C.GeneratedSample("sample", "case", "camera", "P1_AN", "test", "test", "unused", 16, 100, 100, 81,
                               crop_xywh=[0, 0, 100, 100])
    pred = dict(width=100, height=100, fps=16, frame_count=81, frames=[dict(frame=i, detections=[
        dict(track_id=11, label="ball", box=[10, 10, 30, 30]),
        dict(track_id=22, label="ball", box=[70, 10, 90, 30])]) for i in range(81)])
    return gt, sample, pred


def test_upstream_files_unchanged_and_all_cases_defined():
    assert R.code_fingerprint()
    p = G.load_protocol()
    assert len(p["cases"]) == 48
    assert p["settings"]["sam3_fallback"] == "sam2"
    assert p["settings"]["identity_model"] == "facebook/dinov2-base"


def test_global_tracks_are_not_rematched_each_frame(scene):
    gt, sample, pred = scene
    assert E.evaluate(gt, pred, sample)["assignment"] == {"a": 11, "b": 22}
    for fr in pred["frames"][40:]:
        fr["detections"][0]["track_id"], fr["detections"][1]["track_id"] = 22, 11
    result = E.evaluate(gt, pred, sample)
    assert result["summary"]["mean_iou"] == .5
    assert result["summary"]["mean_center_err"] > 0
    assert result["summary"]["n_instants"] == 40
    assert result["alignment"]["source_frames"] == list(range(0, 80, 2))


def test_missing_wrong_class_and_extra_tracks_keep_upstream_semantics(scene):
    gt, sample, pred = scene
    for fr in pred["frames"]:
        fr["detections"][1]["label"] = "car"
        fr["detections"].append(dict(track_id=33, label="ball", box=[40, 60, 60, 80]))
    result = E.evaluate(gt, pred, sample)
    assert result["summary"]["mean_iou"] == .5
    assert result["summary"]["miss_rate"] == .5
    assert result["summary"]["mean_center_err"] == 0  # Missing boxes have no centre error.
    assert result["summary"]["unassigned_tracks"] == [22, 33]


def test_memory_uses_predicted_boxes_and_absence_windows(scene):
    gt, sample, pred = scene
    for fr in gt["frames"][16:40]:
        fr["boxes"]["a"]["visible"] = False
    result = M.evaluate(gt, pred, sample)
    assert result["summary"]["n_events"] == 1
    event = result["events"][0]
    assert event["entity"] == "a" and event["absence"] == 1.625  # Last visible instant to first returned instant.
    assert result["summary"]["reappear_iou"] == 1
    assert result["per_entity"]["a"]["instants"][20]["best_box"] == [.1, .1, .3, .3]


def test_no_identity_pairs_are_na_without_decoding_or_loading_model(scene):
    gt, sample, pred = scene
    for fr in pred["frames"]:
        fr["detections"] = []
    result = R.identity_score("nonexistent.mp4", M.evaluate(gt, pred, sample), sample, None)
    assert result["status"] == "not_applicable"
    assert result["summary"]["identity_short"] is None


def test_source_padding_and_output_mapping(monkeypatch):
    transform = G.letterbox_transform([600, 426], [832, 480])
    assert G.map_source_box([0, 0, 600, 426], transform) == [78, 0, 754, 480]
    monkeypatch.setattr(C, "video_meta", lambda _: dict(width=864, height=480, fps=16, frame_count=81))
    sample = R.sample_for(dict(model="OmniWeaving", case="A01", cell="P1_AN", video="unused"),
                          dict(width=832, height=480), "test")
    assert sample.crop_xywh == pytest.approx([0, (480 - 832 / 1.8) / 2, 832, 832 / 1.8])
    assert sample.spatial.box_src_to_out_norm(sample.crop_xywh[:2] +
        [sample.crop_xywh[0] + sample.crop_xywh[2], sample.crop_xywh[1] + sample.crop_xywh[3]]) == pytest.approx([0, 0, 1, 1])


def test_orthographic_projection_and_negative_depth():
    camera = dict(worldToCamera=np.eye(4).reshape(-1).tolist())
    corners = np.array([[-.5, -.5, -2], [.5, .5, -2]])
    assert G.projected_box(corners, camera, np.eye(4), [100, 100], "ORTHO") == pytest.approx([25, 25, 75, 75])
    corners[:, 2] = 2
    assert G.projected_box(corners, camera, np.eye(4), [100, 100], "ORTHO") is None


def test_exported_entities_have_unique_geometry_and_complete_timeline():
    paths = sorted((GT / "entities").glob("*.json"))
    if not paths:
        pytest.skip("Blender export not present")
    assert len(paths) == 48
    for path in paths:
        gt = json.loads(path.read_text())
        members = [o for e in gt["entities"].values() for o in e["member_objects"]]
        assert len(members) == len(set(members)), path
        assert len(gt["frames"]) == 81
        assert all(set(fr["boxes"]) == set(gt["entities"]) for fr in gt["frames"])
    person = json.loads((paths[0].parent / "C03.json").read_text())
    assert len(person["entities"]) == 2
    assert len(person["entities"]["person"]["member_objects"]) >= 20


def test_new_models_and_vino_mapping(monkeypatch):
    monkeypatch.setattr(C, "video_meta", lambda _: dict(width=832, height=480, fps=16, frame_count=81))
    for model in ("Bernini-R-14B", "LynnReal-Omni", "UniVideo"):
        sample = R.sample_for(dict(model=model, case="A01", cell="P0_AN", video="unused"),
                              dict(width=832, height=480), "test")
        assert sample.crop_xywh == [0, 0, 832, 480]
    sample = R.sample_for(dict(model="VINO", case="A01", cell="P0_AN", video="unused"),
                          dict(width=832, height=480), "test")
    assert sample.crop_xywh[2] < 832 and sample.crop_xywh[3] < 480
