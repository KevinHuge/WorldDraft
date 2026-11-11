"""Report the D-series ViPE camera run without hiding missing videos."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

METRICS = ("RotErr_deg_mean", "TransErr_rel_mean", "CamMC_rel_mean")
GROUPS = ("ALL", "P0", "P1", "P2", "AN", "AA", "AC", *(
    f"P{p}_{a}" for p in range(3) for a in ("AN", "AA", "AC")))


def atomic_text(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def write_csv(path, rows):
    if not rows:
        atomic_text(path, "")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def aggregate(rows, groups=GROUPS):
    """Average within a case, then equally across cases; omit reference self-pairs."""
    output = []
    for model in dict.fromkeys(row["model"] for row in rows):
        model_rows = [row for row in rows if row["model"] == model and not row["reference"]]
        for group in groups:
            selected = [row for row in model_rows if group == "ALL" or
                        group == row["cell"] or group in row["cell"].split("_")]
            if not selected:
                continue
            record = dict(model=model, group=group, expected=len(selected))
            for status in ("complete", "missing", "error", "pending"):
                record[status] = sum(row["status"] == status for row in selected)
            record["available"] = record["expected"] - record["missing"]
            record["PoseValidRate_pct"] = (100 * record["complete"] / record["available"]
                                           if record["available"] else None)
            for metric in METRICS:
                cases = defaultdict(list)
                for row in selected:
                    if row["status"] == "complete" and row[metric] is not None:
                        cases[row["case"]].append(row[metric])
                record[metric] = mean(mean(values) for values in cases.values()) if cases else None
                record[metric + "_cases"] = len(cases)
            output.append(record)
    return output


def collect_rows(out, plan):
    rows = []
    for job in plan["jobs"] + plan.get("missing", []):
        row = {key: job[key] for key in ("model", "case", "cell")}
        row.update(reference=bool(job.get("reference", job["model"] == "draft" and job["cell"] == "P2_AA")),
                   video=job.get("video", job.get("path", "")), status="pending", error="")
        row.update({metric: None for metric in METRICS})
        if job.get("status") == "missing":
            row["status"] = "missing"
        else:
            path = out / "metrics" / job["model"] / job["case"] / f"{job['cell']}.json"
            if path.exists():
                result = json.loads(path.read_text())
                row["status"] = result["status"]
                row["error"] = result.get("error", "")
                row["seconds"] = result.get("seconds", result.get("elapsed_seconds"))
                if row["status"] == "complete":
                    checks = {key: job[key] for key in ("model", "case", "cell")}
                    if plan.get("config_hash"):
                        checks["config_hash"] = plan["config_hash"]
                    if job.get("video_sha256"):
                        checks["input_sha256"] = job["video_sha256"]
                    mismatch = [key for key, value in checks.items() if result.get(key) != value]
                    if mismatch:
                        row.update(status="error", error="Metric provenance mismatch: " + ", ".join(mismatch))
                if row["status"] == "complete":
                    summary = result.get("summary", {})
                    for metric in METRICS:
                        value = summary.get(metric)
                        if value is None or not math.isfinite(float(value)):
                            row.update(status="error", error=f"Missing/non-finite metric: {metric}")
                            break
                        row[metric] = float(value)
                    if row["status"] == "error":
                        row.update({metric: None for metric in METRICS})
        rows.append(row)
    return sorted(rows, key=lambda row: (row["model"] == "draft", row["model"], row["case"], row["cell"]))


def matched_cases(rows, models, complete=False):
    cells = {f"P{p}_{a}" for p in range(3) for a in ("AN", "AA", "AC")}
    found = defaultdict(set)
    for row in rows:
        valid = row["status"] == "complete" if complete else row["status"] != "missing"
        if valid:
            found[row["model"], row["case"]].add(row["cell"])
    return [case for case in sorted({row["case"] for row in rows})
            if all(found[model, case] == cells for model in models)]


def report(out):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    rows = collect_rows(out, plan)
    summaries = aggregate(rows)
    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("complete", "missing", "error", "pending")}
    status = "running" if counts["pending"] else ("complete_with_errors" if counts["error"]
              else "complete_with_missing" if counts["missing"] else "complete")
    workers = [json.loads(path.read_text()) for path in sorted(out.glob("worker-*.json"))]
    comparisons = []
    comparison_cases = {}
    generators = tuple(sorted({row["model"] for row in rows} - {"draft"}))
    for name, models in (("all_generators", generators),):
        for selection, complete in (("available", False), ("complete", True)):
            label = f"{name}_{selection}"
            cases = matched_cases(rows, models, complete=complete)
            comparison_cases[label] = cases
            for summary in aggregate([row for row in rows if row["model"] in models and row["case"] in cases]):
                comparisons.append(dict(comparison=label, cases=";".join(cases), **summary))
    case_summaries = []
    for case in sorted({row["case"] for row in rows}):
        case_summaries.extend(dict(case=case, **row) for row in
                              aggregate([row for row in rows if row["case"] == case]))
    result = dict(status=status, counts=counts, videos=rows, summary=summaries,
                  comparisons=comparisons, comparison_cases=comparison_cases, workers=workers)
    atomic_text(out / "report.json", json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_csv(out / "video_metrics.csv", rows)
    write_csv(out / "model_summary.csv", summaries)
    write_csv(out / "common_case_summary.csv", comparisons)
    write_csv(out / "case_summary.csv", case_summaries)

    def fmt(value):
        return "—" if value is None else f"{value:.4f}"

    def table(selected):
        lines = ["| Model | Cells | Rot (°) ↓ | Trans ↓ | CamMC ↓ | Complete/expected | Missing | Failed |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in selected:
            lines.append(f"| {row['model']} | {row['group']} | " +
                         " | ".join(fmt(row[metric]) for metric in METRICS) +
                         f" | {row['complete']}/{row['expected']} | {row['missing']} | {row['error']} |")
        return lines

    lines = ["# Camera (D series): ViPE", "",
             f"Status: {status}; complete {counts['complete']}, pending {counts['pending']}, "
             f"missing video {counts['missing']}, failed {counts['error']}.", "",
             "ViPE with fixed intrinsics (pose_only) and the CamI2V formulas. Every video of a scene is "
             "compared with the ViPE trajectory of that scene's draft P2_AA at 40 instants (8 Hz, 0-4.875 s). "
             "Rot in degrees, Trans and CamMC unitless; lower is better.", "",
             "Instants are averaged within a video, cells within a scene, then scenes equally. Missing or failed videos "
             "are not counted as zero; read results together with the coverage columns. The draft P2_AA references "
             "compared with themselves are excluded.", "",
             "## Each model on its available scenes", ""]
    lines += table([row for row in summaries if row["group"] == "ALL"])
    for label in ("all_generators_available", "all_generators_complete"):
        lines += ["", f"## {label}: {', '.join(comparison_cases[label]) or 'none'}", ""]
        lines += table([row for row in comparisons if row["comparison"] == label and row["group"] == "ALL"])
    lines += ["", "## P0/P1/P2, AN/AA/AC", ""]
    lines += table([row for row in summaries if row["group"] in GROUPS[1:7]])
    lines += ["", "## Per cell", ""]
    lines += table([row for row in summaries if row["group"] in GROUPS[7:]])
    lines += ["", "Per video: [video_metrics.csv](video_metrics.csv); per scene: [case_summary.csv](case_summary.csv); "
              "scenes shared by all generators: [common_case_summary.csv](common_case_summary.csv).", "",
              "Trajectories are in `poses/<model>/<case>/<cell>.json`, scores and errors in `metrics/` with the same "
              "layout; configuration, video hashes and references in `plan.json`.", "",
              "Trans normalizes each trajectory by its own maximum displacement, so it does not "
              "reflect the overall amount of motion."]
    atomic_text(out / "README.md", "\n".join(lines) + "\n")
    print(json.dumps(dict(status=status, **counts)), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    report(parser.parse_args().out)
