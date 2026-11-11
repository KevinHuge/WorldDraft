"""Projected ground-truth boxes: box from projected corners, clipping to the canvas."""
from __future__ import annotations


import numpy as np

from worlddraft_eval.geometry.common import SOURCE_H, SOURCE_W


def _mat(m):
    """Column-major 16 -> (4,4) row-major numpy."""
    return np.asarray(m, dtype=np.float64).reshape(4, 4).T


def project_points(cam: dict, pts: np.ndarray):
    """Returns (pixels (N,2), depth (N,)) using the recording's intrinsics; camera looks down -Z."""
    W2C = _mat(cam["worldToCamera"])
    k = cam["intrinsics"]
    h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    c = (W2C @ h.T).T[:, :3]
    depth = -c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k["fx"] * c[:, 0] / depth + k["cx"]
        v = k["cy"] - k["fy"] * c[:, 1] / depth
    return np.stack([u, v], axis=1), depth


def box_from_corners(cam: dict, corners: np.ndarray, near_eps: float = 1e-3):
    """Pixel AABB of the projected corners. Corners behind the camera are clipped
    against the near plane along the edges to the visible corners (cheap and good
    enough for a bbox)."""
    pix, depth = project_points(cam, corners)
    front = depth > near_eps
    if not front.any():
        return None, 0
    if not front.all():
        W2C = _mat(cam["worldToCamera"]); k = cam["intrinsics"]
        cc = (W2C @ np.concatenate([corners, np.ones((8, 1))], 1).T).T[:, :3]
        edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
        extra = []
        for i, j in edges:
            if front[i] != front[j]:
                a, b = cc[i], cc[j]
                za, zb = -a[2], -b[2]
                t = (near_eps - za) / (zb - za)
                p = a + t * (b - a)
                d = -p[2]
                extra.append([k["fx"] * p[0] / d + k["cx"], k["cy"] - k["fy"] * p[1] / d])
        pix = np.concatenate([pix[front], np.asarray(extra).reshape(-1, 2)], axis=0)
    else:
        pix = pix[front]
    x0, y0 = pix.min(0); x1, y1 = pix.max(0)
    return [float(x0), float(y0), float(x1), float(y1)], int(front.sum())


def clip_box(b, w=SOURCE_W, h=SOURCE_H):
    x0, y0, x1, y1 = b
    x0, x1 = max(0.0, x0), min(float(w), x1)
    y0, y1 = max(0.0, y0), min(float(h), y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


