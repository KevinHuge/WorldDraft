"""Interaction scoring of judge answers: action and outcome per event."""
from __future__ import annotations
import numpy as np


def composite(frame_rgb: np.ndarray, box_src, src_wh, tile_h: int = 360):
    """full frame | zoom, both `tile_h` high. `box_src` is in source pixels; frame may have another size."""
    import cv2
    h, w = frame_rgb.shape[:2]
    sx, sy = w / src_wh[0], h / src_wh[1]
    x0, y0, x1, y1 = box_src
    crop = frame_rgb[int(y0 * sy):max(int(y0 * sy) + 1, int(y1 * sy)), int(x0 * sx):max(int(x0 * sx) + 1, int(x1 * sx))]
    full = cv2.resize(frame_rgb, (int(w * tile_h / h), tile_h), interpolation=cv2.INTER_AREA)
    ch, cw = crop.shape[:2]
    zoom = cv2.resize(crop, (max(1, int(cw * tile_h / ch)), tile_h), interpolation=cv2.INTER_CUBIC)
    sep = np.full((tile_h, 6, 3), 255, dtype=np.uint8)
    return np.concatenate([full, sep, zoom], axis=1)


def score_sample(results: list[dict]) -> dict:
    n = len(results); valid = [r for r in results if not r["invalid"]]
    def rate(k, v="yes"):
        return sum(r["answers"][k] == v for r in valid) / n if n else None
    return dict(interaction_success=sum(r["correct"] for r in valid) / n if n else None, n_events=n,
                n_correct=sum(r["correct"] for r in valid), n_invalid=n - len(valid),
                action_rate=rate("action"), outcome_rate=rate("outcome"), participants_rate=rate("participants"),
                unknown_action_rate=rate("action", "unknown"), unknown_outcome_rate=rate("outcome", "unknown"))
