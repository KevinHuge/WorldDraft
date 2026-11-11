"""ViPE pose estimation (pose-only pipeline) and camera trajectory errors: rotation, scale-normalized translation, CamMC."""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

EVAL_DURATION = 5.0
TRAJ_SCHEMA = "worlddraft.traj"
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
VIPE_PIPELINES = {"pose_only": ["pipeline=default", "pipeline.post.depth_align_model=null"]}

def positive_depth_summary(depths, min_points=32):
    values = np.asarray(depths, float).reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < min_points:
        raise ValueError(f"Scene scale unavailable: {len(values)} valid background points, need {min_points}")
    return dict(depth=float(np.median(values)), n_points=len(values),
                p10=float(np.percentile(values, 10)), p90=float(np.percentile(values, 90)))


def slam_scene_scale(xyz_world, c2w_cv):
    """Same-map camera-Z depth; world frame and global reconstruction scale cancel."""
    xyz = np.asarray(xyz_world, float)
    pose = np.asarray(c2w_cv, float)
    camera_points = (xyz - pose[:3, 3]) @ pose[:3, :3]
    return dict(positive_depth_summary(camera_points[:, 2]), frame=0,
                source="vipe_first_keyframe_filtered_static_slam_map",
                shares_trajectory_scale=True)


def rotmat_to_quat_xyzw(R: np.ndarray) -> list[float]:
    """Shepperd's method; returns [x, y, z, w] with w >= 0."""
    m = np.asarray(R, float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w], float)
    if w < 0:
        q = -q
    return (q / np.linalg.norm(q)).tolist()


def c2w_cv_to_traj(c2w_cv: np.ndarray, fps: float, estimator: dict) -> dict:
    """(N,4,4) OpenCV camera-to-world -> worlddraft.traj (OpenGL convention, per-frame)."""
    frames = []
    for i, M in enumerate(np.asarray(c2w_cv, float)):
        G = M @ CV_TO_GL
        frames.append(dict(frame=i, position=G[:3, 3].tolist(), quaternion=rotmat_to_quat_xyzw(G[:3, :3])))
    return dict(schema=TRAJ_SCHEMA, fps=float(fps), convention="camera_to_world, OpenGL (y up, looks -z), xyzw",
                frames=frames, estimator=estimator)


class VipeEstimator:
    """ViPE camera estimation with fixed intrinsics and masks on the dynamic entities."""

    def __init__(self, vipe_root: str, device: str = "cuda:0", vipe_pipeline: str = "pose_only"):
        self.vipe_root = str(Path(vipe_root).resolve())
        if self.vipe_root not in sys.path:
            sys.path.insert(0, self.vipe_root)
        import torch
        self.torch = torch
        torch.cuda.set_device(torch.device(device))
        from vipe.config.parse import parse_typed_config
        from vipe.pipeline import _as_dictconfig
        from vipe.pipeline.default import DefaultAnnotationPipeline
        from vipe.pipeline.processors import TrackAnythingProcessor
        from vipe.streams.base import FrameAttribute, ProcessedVideoStream, StreamProcessor
        from vipe.utils.cameras import CameraType
        import copy

        typed = parse_typed_config(
            config_name="default", config_dir=str(Path(self.vipe_root) / "configs"),
            hydra_args=VIPE_PIPELINES.get(vipe_pipeline, [f"pipeline={vipe_pipeline}"]) + [
                        "streams=raw_mp4_stream", "streams.base_path=unused",
                        "pipeline.init.intrinsics=gt", "pipeline.output.save_viz=false",
                        "pipeline.output.path=" + str(Path(os.environ.get("TMPDIR", "/tmp")) / "worlddraft_vipe_unused")])
        pcfg = copy.deepcopy(_as_dictconfig(typed.pipeline))
        del pcfg.instance
        self.version = _vipe_version(self.vipe_root)
        self.pipeline_name = vipe_pipeline

        class FixedIntrinsicsProcessor(StreamProcessor):
            def __init__(self, K):
                super().__init__()
                self.K = torch.as_tensor(K).float()

            def update_attributes(self, previous_attributes):
                return previous_attributes | {FrameAttribute.INTRINSICS}

            def __call__(self, frame_idx, frame):
                frame.intrinsics = self.K.clone()
                frame.camera_type = CameraType.PINHOLE
                return frame

        outer = self

        class PoseOnlyPipeline(DefaultAnnotationPipeline):
            def _add_init_processors(self, video_stream):
                procs = [FixedIntrinsicsProcessor(outer._K)]
                if self.init_cfg.instance is not None:
                    procs.append(TrackAnythingProcessor(
                        outer._phrases, add_sky=self.init_cfg.instance.add_sky,
                        sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                        model_cache=self.model_cache))
                return ProcessedVideoStream(video_stream, procs)

        self.pipeline = PoseOnlyPipeline(**pcfg)
        self.pipeline.return_payload = True
        self._K, self._phrases = None, None

    def estimate(self, video: str, K: list[float], phrases: list[str], name: str) -> tuple[np.ndarray, dict]:
        from vipe.streams.raw_mp4_stream import RawMp4Stream
        self._K, self._phrases = K, phrases
        stream = RawMp4Stream(Path(video), name=name)
        n = len(stream)
        t0 = time.time()
        out = self.pipeline.run(stream)
        slam = out.payload
        traj = slam.get_view_trajectory(0)
        c2w = traj.matrix().detach().cpu().numpy()
        if c2w.shape[0] != n:
            raise RuntimeError(f"ViPE returned {c2w.shape[0]} poses for {n} frames ({video})")
        intr = slam.intrinsics[0].detach().cpu().numpy().tolist()
        info = dict(name="vipe", version=self.version, pipeline=self.pipeline_name, intrinsics_fixed=[float(k) for k in K],
                    intrinsics_slam=intr, phrases=list(phrases), ba_residual=float(slam.ba_residual),
                    frames=int(n), fps=float(stream.fps()), seconds=round(time.time() - t0, 1))
        try:
            if slam.slam_map is None or 0 not in slam.slam_map.dense_disp_frame_inds:
                raise ValueError("ViPE has no frame-zero SLAM map")
            index = slam.slam_map.dense_disp_frame_inds.index(0)
            xyz, _ = slam.slam_map.get_dense_disp_pcd(index, 0)
            info["scene_scale"] = slam_scene_scale(xyz.detach().cpu().numpy(), c2w[0])
        except ValueError as exc:
            info["scene_scale_error"] = str(exc)
        self.torch.cuda.empty_cache()
        return c2w, info


def _vipe_version(root: str) -> str:
    try:
        import subprocess
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True,
                                      stderr=subprocess.DEVNULL).strip()
    except Exception:
        rev = "unknown"
    try:
        import vipe
        v = getattr(vipe, "__version__", None) or "?"
    except Exception:
        v = "?"
    return f"{v}@{rev}"


def quat_to_R(q):
    if np.asarray(q).shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid quaternion: expected four finite components and nonzero norm")
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _sample_traj(traj: dict, instants, fps=None, frame_count=None):
    frames = traj["frames"]
    if not frames:
        raise ValueError("Empty camera trajectory")
    if "time" in frames[0]:
        times = np.asarray([fr["time"] for fr in frames], float)
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("Camera timestamps must be finite and strictly increasing")
        tolerance = 1e-9
    else:
        rate = fps or traj.get("fps")
        if rate is None or not np.isfinite(rate) or rate <= 0:
            raise ValueError("Camera trajectory needs a positive finite FPS")
        if fps is not None and traj.get("fps") is not None and not np.isclose(fps, traj['fps'], rtol=1e-5):
            raise ValueError("Prediction FPS disagrees with trajectory metadata")
        if [fr.get("frame", i) for i, fr in enumerate(frames)] != list(range(len(frames))):
            raise ValueError("Frame-index trajectory must be contiguous from frame zero")
        times = np.arange(len(frames)) / rate
        tolerance = .5 / rate + 1e-9
    if min(instants) < times[0] - tolerance or max(instants) > times[-1] + tolerance:
        raise ValueError("Camera trajectory does not cover the evaluation time grid")
    P, Q = [], []
    for t in instants:
        if "time" in frames[0]:
            i = int(np.argmin(abs(times - t)))
        else:
            f = output_frame_for_time(t, rate)
            f = min(f, (frame_count or len(frames)) - 1)
            i = min(f, len(frames) - 1)
        P.append(frames[i]["position"]); Q.append(frames[i]["quaternion"])
    return np.asarray(P, float), np.asarray(Q, float)


def _c2w(P, Q):
    M = np.tile(np.eye(4), (len(P), 1, 1))
    for i in range(len(P)):
        M[i, :3, :3] = quat_to_R(Q[i]); M[i, :3, 3] = P[i]
    return M


def cami2v_metrics(Pg, Qg, Pp, Qp, static: bool = False, eps: float = 1e-9,
                   return_per_instant: bool = False):
    """RotErr / TransErr / CamMC exactly as CamI2V's `metric()`, on relative-to-first-instant c2w poses."""
    G, E = _c2w(Pg, Qg), _c2w(Pp, Qp)
    G = np.linalg.inv(G[0])[None] @ G
    E = np.linalg.inv(E[0])[None] @ E
    n = len(G)
    Rg, Re = G[:, :3, :3], E[:, :3, :3]
    tr = np.einsum("nij,nij->n", Rg, Re)
    rot = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    out = dict(RotErr=float(rot.sum()), RotErr_mean=float(rot.mean()))
    rows = [dict(RotErr=float(r)) for r in rot]
    sg = np.linalg.norm(G[:, :3, 3], axis=1).max() + eps
    se = np.linalg.norm(E[:, :3, 3], axis=1).max() + eps
    for tag, s_e in (("rel", se), ("abs", sg)):
        tg, te = G[:, :3, 3] / sg, E[:, :3, 3] / s_e
        terr = np.linalg.norm(tg - te, axis=1)
        RTg = np.concatenate([Rg, tg[:, :, None]], axis=2).reshape(n, 12)
        RTe = np.concatenate([Re, te[:, :, None]], axis=2).reshape(n, 12)
        mc = np.linalg.norm(RTg - RTe, axis=1)
        out[f"TransErr_{tag}"] = float(terr.sum()); out[f"TransErr_{tag}_mean"] = float(terr.mean())
        out[f"CamMC_{tag}"] = float(mc.sum()); out[f"CamMC_{tag}_mean"] = float(mc.mean())
        for row, t, c in zip(rows, terr, mc):
            row[f"TransErr_{tag}"] = float(t)
            row[f"CamMC_{tag}"] = float(c)
    return (out, rows) if return_per_instant else out


def output_frame_for_time(t: float, fps: float) -> int:
    return int(round(t * fps))


def eval_grid(grid_hz: float = 8.0, duration: float = EVAL_DURATION) -> list[float]:
    """Common evaluation instants for every generator: k / grid_hz, k>=0, t<duration.
    8 Hz -> 40 instants on a 5 s clip. Lower than every generator's fps, so we
    never interpolate: each instant picks the nearest generated frame and the
    nearest source frame."""
    if not np.isfinite([grid_hz, duration]).all() or grid_hz <= 0 or duration <= 0:
        raise ValueError("Evaluation grid frequency and duration must be positive and finite")
    n = int(math.floor(duration * grid_hz - 1e-9)) + 1
    return [k / grid_hz for k in range(n) if k / grid_hz < duration - 1e-9]

