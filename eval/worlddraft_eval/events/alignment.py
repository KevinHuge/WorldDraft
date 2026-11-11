"""Timeline-alignment scoring of judge answers."""


def _rates(answers: list[str]) -> dict:
    n = len(answers)
    if n == 0:
        return dict(score=None, n=0, n_yes=0, n_unknown=0, n_invalid=0)
    return dict(score=sum(a == "yes" for a in answers) / n, n=n, n_yes=sum(a == "yes" for a in answers),
                n_unknown=sum(a == "unknown" for a in answers), n_invalid=sum(a == "invalid" for a in answers))


def score_sample(segments, end_state, static, foil_segments=None, foil_static=None, judge_name=None) -> dict:
    d = _rates([s["depicted"] for s in segments]); e = _rates(end_state["answers"]); st = _rates(static["answers"])
    strict = _rates([(s.get("details") or "invalid") if s["depicted"] == "yes" else s["depicted"] for s in segments])
    out = dict(dynamic_checklist=d["score"], dynamic_strict=strict["score"], n_segments=d["n"],
               dynamic_unknown_rate=d["n_unknown"] / d["n"] if d["n"] else None,
               end_state=e["score"], n_end_facts=e["n"],
               static_facts=st["score"], n_static_facts=st["n"], static_unknown_rate=st["n_unknown"] / st["n"] if st["n"] else None,
               n_invalid=d["n_invalid"] + e["n_invalid"] + st["n_invalid"], judge=judge_name)
    if foil_segments is not None:
        fd = _rates([s["depicted"] for s in foil_segments])
        out.update(dynamic_foil=fd["score"], dynamic_margin=(d["score"] - fd["score"]) if None not in (d["score"], fd["score"]) else None)
        out["n_invalid"] += fd["n_invalid"]
    if foil_static is not None:
        fs = _rates(foil_static["answers"])
        out.update(static_foil=fs["score"], static_margin=(st["score"] - fs["score"]) if None not in (st["score"], fs["score"]) else None)
        out["n_invalid"] += fs["n_invalid"]
    return out
