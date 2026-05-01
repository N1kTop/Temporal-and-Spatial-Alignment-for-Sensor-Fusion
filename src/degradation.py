"""
degradation.py

Utilities for creating controlled spatial and temporal degradation.

This file contains the corruption functions used in the alignment experiments:
- spatial translation and yaw errors
- frame-level temporal offsets
- timestamp jitter, dropout, and drift
- per-sensor corruption maps
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .data_utils import get_scene_samples


NumberLike = Union[int, float]


# --------------------------------------------------
# Spatial degradation
# --------------------------------------------------

def apply_spatial_transform(
    points_xyz: np.ndarray,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """
    Apply a simple rigid transform to radar points already expressed
    in the LiDAR/reference frame.

    The transform is:
        1. yaw rotation around the z-axis
        2. translation in x, y, z

    points_xyz should have shape (3, N).
    """
    pts = points_xyz.copy()

    theta = np.deg2rad(yaw_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    R = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    pts = R @ pts

    pts[0, :] += dx
    pts[1, :] += dy
    pts[2, :] += dz

    return pts


def inverse_spatial_transform_params(
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
) -> Dict[str, float]:
    """
    Approximate inverse parameters for small controlled experiments.

    For the experiments in this project, corrections were normally tested
    by applying the negative translation / yaw values as candidate corrections.
    """
    return {
        "dx": -dx,
        "dy": -dy,
        "dz": -dz,
        "yaw_deg": -yaw_deg,
    }


def spatial_degradation_configs(
    dx_values: Optional[Sequence[float]] = None,
    yaw_values: Optional[Sequence[float]] = None,
) -> List[Dict[str, float]]:
    """
    Build a small list of spatial degradation configurations.

    Useful when running repeated experiments in notebooks.
    """
    configs: List[Dict[str, float]] = []

    if dx_values is not None:
        for dx in dx_values:
            configs.append({
                "label": f"dx_{dx}m",
                "dx": float(dx),
                "dy": 0.0,
                "dz": 0.0,
                "yaw_deg": 0.0,
            })

    if yaw_values is not None:
        for yaw in yaw_values:
            configs.append({
                "label": f"yaw_{yaw}deg",
                "dx": 0.0,
                "dy": 0.0,
                "dz": 0.0,
                "yaw_deg": float(yaw),
            })

    return configs


# --------------------------------------------------
# Frame-level temporal degradation
# --------------------------------------------------

def get_temporal_offset_pairs(
    samples: Sequence[dict],
    frame_offset: int = 0,
) -> List[Tuple[int, int, dict, dict]]:
    """
    Create LiDAR/radar sample pairs using a frame offset.

    frame_offset:
        0  = aligned baseline
        -1 = radar from previous frame
        +1 = radar from next frame

    Returns:
        list of (lidar_index, radar_index, lidar_sample, radar_sample)
    """
    pairs = []

    for i, lidar_sample in enumerate(samples):
        radar_index = i + frame_offset

        if radar_index < 0 or radar_index >= len(samples):
            continue

        pairs.append((i, radar_index, lidar_sample, samples[radar_index]))

    return pairs


def get_temporal_offset_pairs_for_scene(
    nusc,
    scene_index: int = 0,
    frame_offset: int = 0,
) -> List[Tuple[int, int, dict, dict]]:
    """
    Convenience wrapper for frame-offset pairing across a scene.
    """
    samples = get_scene_samples(nusc, scene_index=scene_index)
    return get_temporal_offset_pairs(samples, frame_offset=frame_offset)


# --------------------------------------------------
# Timestamp corruption functions
# --------------------------------------------------

def drift_offset_fn(
    a_ms_per_frame: float = 0.5,
    b_ms: float = 0.0,
    jitter_ms: float = 0.0,
    seed: int = 42,
) -> Callable[[int], float]:
    """
    Return a function that gives timestamp offset in ms for each frame.

    offset(i) = b + a*i + random_jitter

    This mirrors the drift-style corruption used in the notebook.
    """
    rng = np.random.default_rng(seed)

    def fn(i: int) -> float:
        drift = b_ms + a_ms_per_frame * i
        noise = rng.normal(0.0, jitter_ms) if jitter_ms > 0 else 0.0
        return float(drift + noise)

    return fn


def apply_timestamp_offset(
    timestamps_us: np.ndarray,
    offset_ms: Union[NumberLike, Callable[[int], float]],
) -> np.ndarray:
    """
    Apply either a constant or frame-dependent timestamp offset.

    timestamps are in microseconds.
    offset is in milliseconds.
    """
    ts = np.asarray(timestamps_us, dtype="float64").copy()

    for i in range(len(ts)):
        if not np.isfinite(ts[i]):
            continue

        off_ms = offset_ms(i) if callable(offset_ms) else offset_ms
        ts[i] += float(off_ms) * 1000.0

    return ts


def corrupt_timestamps(
    timestamps_us: np.ndarray,
    jitter_ms: float = 0.0,
    drop_prob: float = 0.0,
    drift_ms_per_frame: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Corrupt a single timestamp sequence using dropout, drift, and jitter.

    Order:
        1. randomly drop timestamps by setting them to NaN
        2. apply linear drift
        3. apply Gaussian jitter
    """
    rng = np.random.default_rng(seed)

    ts = np.asarray(timestamps_us, dtype="float64").copy()
    n = len(ts)
    frame_idx = np.arange(n, dtype="float64")

    valid = np.isfinite(ts)

    if drop_prob > 0:
        drop_mask = rng.random(n) < drop_prob
        ts[valid & drop_mask] = np.nan

    valid = np.isfinite(ts)

    if drift_ms_per_frame != 0.0:
        ts[valid] += drift_ms_per_frame * frame_idx[valid] * 1000.0

    if jitter_ms > 0:
        ts[valid] += rng.normal(0.0, jitter_ms * 1000.0, size=int(valid.sum()))

    return ts


def corrupt_df_timestamps(
    df: pd.DataFrame,
    sensors: Sequence[str],
    jitter_ms: float = 0.0,
    drop_prob: float = 0.0,
    drift_ms_per_frame: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Corrupt selected sensor timestamp columns in a timestamp table.

    The same corruption settings are applied to each selected sensor.
    """
    out = df.copy(deep=True)

    for k, sensor in enumerate(sensors):
        if sensor not in out.columns:
            continue

        out[sensor] = corrupt_timestamps(
            out[sensor].to_numpy(dtype="float64"),
            jitter_ms=jitter_ms,
            drop_prob=drop_prob,
            drift_ms_per_frame=drift_ms_per_frame,
            seed=seed + k,
        )

    return out


def apply_drift_plus_jitter(
    df: pd.DataFrame,
    sensors: Sequence[str],
    drift_ms_per_frame_map: Dict[str, float],
    jitter_ms_map: Optional[Dict[str, float]] = None,
    bursty_radar: bool = True,
    burst_every: int = 20,
    burst_len: int = 4,
    burst_extra_jitter_ms: float = 40.0,
    seed: int = 123,
) -> pd.DataFrame:
    """
    Apply per-sensor drift and jitter to timestamp columns.

    This version is closer to the more detailed notebook experiment:
    - cameras can have smaller jitter
    - radars can have larger jitter
    - optional radar burst noise can be added
    """
    rng = np.random.default_rng(seed)
    out = df.copy(deep=True)

    n = len(out)
    frame_idx = np.arange(n, dtype="float64")

    if jitter_ms_map is None:
        jitter_ms_map = {s: 0.0 for s in sensors}

    for sensor in sensors:
        if sensor not in out.columns:
            continue

        ts = out[sensor].to_numpy(dtype="float64").copy()
        valid = np.isfinite(ts)

        drift = drift_ms_per_frame_map.get(sensor, 0.0)
        if drift != 0.0:
            ts[valid] += drift * frame_idx[valid] * 1000.0

        jitter = jitter_ms_map.get(sensor, 0.0)
        if jitter > 0:
            ts[valid] += rng.normal(0.0, jitter * 1000.0, size=int(valid.sum()))

        if bursty_radar and sensor.startswith("RADAR") and burst_extra_jitter_ms > 0:
            burst_mask = np.zeros(n, dtype=bool)

            for start in range(0, n, burst_every):
                burst_mask[start:start + burst_len] = True

            burst_mask &= np.isfinite(ts)

            ts[burst_mask] += rng.normal(
                0.0,
                burst_extra_jitter_ms * 1000.0,
                size=int(burst_mask.sum()),
            )

        out[sensor] = ts

    return out


# --------------------------------------------------
# Default corruption settings from the notebook style
# --------------------------------------------------

def default_jitter_config(sensors: Iterable[str]) -> Dict[str, float]:
    """
    Simple default jitter config:
    - cameras: lower jitter
    - radar: higher jitter
    """
    cfg = {}

    for sensor in sensors:
        cfg[sensor] = 5.0 if sensor.startswith("CAM") else 12.0

    return cfg


def default_drift_config() -> Dict[str, float]:
    """
    Drift settings similar to the exploratory notebook.

    Values are in ms/frame.
    """
    return {
        "CAM_FRONT": 0.10,
        "CAM_FRONT_LEFT": 0.08,
        "CAM_FRONT_RIGHT": 0.08,
        "CAM_BACK": 0.06,
        "CAM_BACK_LEFT": 0.06,
        "CAM_BACK_RIGHT": 0.06,

        "RADAR_FRONT": 0.30,
        "RADAR_FRONT_LEFT": 0.25,
        "RADAR_FRONT_RIGHT": 0.25,
        "RADAR_BACK_LEFT": 0.20,
        "RADAR_BACK_RIGHT": 0.20,
    }


# --------------------------------------------------
# Small utilities
# --------------------------------------------------

def describe_spatial_transform(
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
) -> str:
    """
    Return a short readable description of a spatial transform.
    """
    return f"dx={dx:.3f} m, dy={dy:.3f} m, dz={dz:.3f} m, yaw={yaw_deg:.3f} deg"


def make_condition_name(
    dx: float = 0.0,
    dy: float = 0.0,
    yaw_deg: float = 0.0,
    frame_offset: Optional[int] = None,
) -> str:
    """
    Make a simple label for plots/tables.
    """
    if frame_offset is not None and frame_offset != 0:
        return f"frame_offset_{frame_offset:+d}"

    if abs(yaw_deg) > 0:
        return f"yaw_{yaw_deg:g}deg"

    if abs(dx) > 0 or abs(dy) > 0:
        return f"dx_{dx:g}m_dy_{dy:g}m"

    return "baseline"


if __name__ == "__main__":
    print("degradation.py contains spatial and temporal degradation helpers.")