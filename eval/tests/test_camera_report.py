import json

from worlddraft_eval.camera.report import METRICS, aggregate, collect_rows, matched_cases


def row(case, cell, score, status="complete", model="Wan2.1-VACE-14B", reference=False):
    return dict(model=model, case=case, cell=cell, status=status, reference=reference,
                **{metric: score for metric in METRICS})


def test_case_equal_mean_and_reference_exclusion():
    rows = [row("D01", "P0_AA", 2), row("D01", "P1_AA", 4), row("D02", "P0_AA", 9),
            row("D03", "P0_AA", None, "missing"), row("D04", "P0_AA", None, "error")]
    result = aggregate(rows, ("ALL",))[0]
    assert result[METRICS[0]] == 6
    assert result["missing"] == 1 and result["error"] == 1
    assert result["PoseValidRate_pct"] == 75
    controls = [row("D01", "P2_AA", 0, model="draft", reference=True),
                row("D01", "P1_AA", 6, model="draft")]
    assert aggregate(controls, ("ALL",))[0][METRICS[0]] == 6


def test_missing_and_nonfinite_never_become_good_scores(tmp_path):
    base = dict(model="Wan2.1-VACE-14B", case="D01", cell="P0_AA", reference=False, video="input.mp4")
    path = tmp_path / "metrics/Wan2.1-VACE-14B/D01/P0_AA.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(base, status="complete", summary={metric: None for metric in METRICS})))
    missing = dict(base, cell="P1_AA", status="missing")
    results = collect_rows(tmp_path, dict(jobs=[base], missing=[missing]))
    assert [r["status"] for r in results] == ["error", "missing"]
    assert all(r[METRICS[0]] is None for r in results)


def test_old_metric_hash_is_not_reported_as_current(tmp_path):
    job = dict(model="Wan2.1-VACE-14B", case="D01", cell="P0_AA", video_sha256="new-video")
    path = tmp_path / "metrics/Wan2.1-VACE-14B/D01/P0_AA.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(job, status="complete", input_sha256="old-video", config_hash="config",
                                    summary={metric: 0 for metric in METRICS})))
    result = collect_rows(tmp_path, dict(jobs=[job], missing=[], config_hash="config"))[0]
    assert result["status"] == "error"
    assert "input_sha256" in result["error"]
    assert result[METRICS[0]] is None


def test_common_case_requires_all_cells_for_each_model():
    rows = [row("D07", f"P{p}_{a}", 1, model=model)
            for model in ("MiniMax", "Wan2.1-VACE-14B") for p in range(3) for a in ("AN", "AA", "AC")]
    rows[-1]["status"] = "error"
    assert matched_cases(rows, ("MiniMax", "Wan2.1-VACE-14B")) == ["D07"]
    assert matched_cases(rows, ("MiniMax", "Wan2.1-VACE-14B"), complete=True) == []


def test_new_inventory_model_can_be_collected(tmp_path):
    job = dict(model="UniVideo", case="D01", cell="P0_AN", status="missing")
    result = collect_rows(tmp_path, dict(jobs=[], missing=[job]))
    assert result[0]["model"] == "UniVideo"
    assert result[0]["status"] == "missing"
