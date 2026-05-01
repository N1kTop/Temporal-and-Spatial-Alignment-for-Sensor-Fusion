"""
multiframe.py

Multiframe LiDAR-radar fusion utilities.

This file contains the temporal aggregation methods used in the project:
- naive multiframe fusion
- temporal-weighted multiframe fusion
- simple radar motion compensation
- comparison helpers for baseline/degraded/multiframe results

The frame format matches fusion.py:

    {
        "frame_index": int,
        "timestamp": int,
        "objects": [...]
    }
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import points_in_box

from .data_utils import get_scene_samples, get_adjacent_samples
from .degradation import apply_spatial_transform
from .fusion import (
    compute_weighted_velocity,
    run_scene_fusion,
    run_scene_temporal_offset,
)
from .metrics import summarise_results, compute_physical_alignment_metrics


# --------------------------------------------------
# Small radar helpers
# --------------------------------------------------

def load_radar_xyz_velocity(
    nusc,
    sample: dict,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load radar points and compensated velocity.

    Returns:
        radar_xyz: (3, N)
        radar_vel_xy: (2, N)
    """
    radar_pc, _ = RadarPointCloud.from_file_multisweep(
        nusc,
        sample,
        chan=radar_chan,
        ref_chan=ref_chan,
        nsweeps=1,
    )

    radar_xyz = radar_pc.points[:3, :].copy()
    radar_vel_xy = np.vstack((radar_pc.points[8, :], radar_pc.points[9, :]))

    radar_xyz = apply_spatial_transform(
        radar_xyz,
        dx=dx,
        dy=dy,
        dz=dz,
        yaw_deg=yaw_deg,
    )

    return radar_xyz, radar_vel_xy


def get_lidar_boxes(
    nusc,
    sample: dict,
    ref_chan: str = "LIDAR_TOP",
):
    """
    Load LiDAR/reference boxes for one sample.
    """
    lidar_sd = nusc.get("sample_data", sample["data"][ref_chan])
    _, boxes, _ = nusc.get_sample_data(lidar_sd["token"])
    return boxes


# --------------------------------------------------
# Naive multiframe fusion
# --------------------------------------------------

def get_multiframe_samples(
    samples: Sequence[dict],
    frame_index: int,
    frame_window: int = 1,
) -> List[Tuple[int, dict]]:
    """
    Return neighbouring samples around frame_index.

    frame_window = 1 gives:
        t-1, t, t+1

    Returns:
        list of (offset, sample)
    """
    selected = []

    for offset in range(-frame_window, frame_window + 1):
        idx = frame_index + offset

        if idx < 0 or idx >= len(samples):
            continue

        selected.append((offset, samples[idx]))

    return selected


def extract_multiframe_measurements(
    nusc,
    samples: Sequence[dict],
    frame_index: int,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    min_hits: int = 1,
    max_range: float = 60.0,
    frame_window: int = 1,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
) -> List[Dict]:
    """
    Naive multiframe fusion.

    Radar points from neighbouring frames are concatenated first, then
    associated with current LiDAR boxes.
    """
    curr_sample = samples[frame_index]
    boxes = get_lidar_boxes(nusc, curr_sample, ref_chan=ref_chan)

    all_points = []
    all_velocities = []

    for _, radar_sample in get_multiframe_samples(samples, frame_index, frame_window):
        radar_xyz, radar_vel_xy = load_radar_xyz_velocity(
            nusc,
            radar_sample,
            radar_chan=radar_chan,
            ref_chan=ref_chan,
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_deg=yaw_deg,
        )

        all_points.append(radar_xyz)
        all_velocities.append(radar_vel_xy)

    if not all_points:
        return []

    radar_xyz = np.hstack(all_points)
    radar_vel_xy = np.hstack(all_velocities)

    objects = []

    for box_index, box in enumerate(boxes):
        if np.linalg.norm(box.center[:2]) > max_range:
            continue

        inside = points_in_box(box, radar_xyz)
        hit_count = int(np.sum(inside))

        if hit_count < min_hits:
            continue

        points_xy = radar_xyz[:2, inside].T
        velocities_xy = radar_vel_xy[:, inside].T
        centre_xy = box.center[:2].copy()

        velocity_estimate = compute_weighted_velocity(
            points_xy,
            velocities_xy,
            centre_xy,
        )

        objects.append({
            "box_index": box_index,
            "box_name": box.name,
            "position_xy": centre_xy,
            "velocity_xy": velocity_estimate,
            "radar_hits": hit_count,
            "associated_radar_points_xy": points_xy.tolist(),
        })

    return objects


def run_scene_multiframe_fusion(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    min_hits: int = 1,
    max_range: float = 60.0,
    frame_window: int = 1,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
) -> List[Dict]:
    """
    Run naive multiframe fusion across a scene.
    """
    samples = get_scene_samples(nusc, scene_index=scene_index)
    frames = []

    for frame_index, sample in enumerate(samples):
        objects = extract_multiframe_measurements(
            nusc,
            samples,
            frame_index,
            radar_chan=radar_chan,
            ref_chan=ref_chan,
            min_hits=min_hits,
            max_range=max_range,
            frame_window=frame_window,
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_deg=yaw_deg,
        )

        frames.append({
            "frame_index": frame_index,
            "timestamp": sample["timestamp"],
            "objects": objects,
        })

    return frames


# --------------------------------------------------
# Temporal-weighted multiframe fusion
# --------------------------------------------------

def temporal_weight(frame_offset: int, alpha: float = 0.7) -> float:
    """
    Exponential temporal weight.

    Current frame has offset 0 and weight 1.
    Neighbouring frames have lower weight.
    """
    return float(np.exp(-alpha * abs(frame_offset)))


def compute_temporal_weighted_velocity(
    points_xy: np.ndarray,
    radar_vel_xy: np.ndarray,
    box_center_xy: np.ndarray,
    frame_offsets: np.ndarray,
    alpha: float = 0.7,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Combine spatial weighting and temporal weighting.

    Spatial:
        points closer to the box centre get larger weights

    Temporal:
        points from the current frame get larger weights than neighbours
    """
    if len(points_xy) == 0:
        return np.array([0.0, 0.0])

    dists = np.linalg.norm(points_xy - box_center_xy[None, :], axis=1)
    spatial_weights = 1.0 / (dists + eps)

    temporal_weights = np.exp(-alpha * np.abs(frame_offsets))
    weights = spatial_weights * temporal_weights

    if np.sum(weights) <= 0:
        return np.mean(radar_vel_xy, axis=0)

    return np.sum(radar_vel_xy * weights[:, None], axis=0) / np.sum(weights)


def extract_temporal_weighted_multiframe_measurements(
    nusc,
    samples: Sequence[dict],
    frame_index: int,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    min_hits: int = 1,
    max_range: float = 60.0,
    frame_window: int = 1,
    alpha: float = 0.7,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
) -> List[Dict]:
    """
    Temporal-weighted multiframe fusion.

    Similar to naive multiframe fusion, but velocity is weighted by:
    - spatial distance from the object centre
    - temporal distance from the current frame
    """
    curr_sample = samples[frame_index]
    boxes = get_lidar_boxes(nusc, curr_sample, ref_chan=ref_chan)

    all_points = []
    all_velocities = []
    all_offsets = []

    for offset, radar_sample in get_multiframe_samples(samples, frame_index, frame_window):
        radar_xyz, radar_vel_xy = load_radar_xyz_velocity(
            nusc,
            radar_sample,
            radar_chan=radar_chan,
            ref_chan=ref_chan,
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_deg=yaw_deg,
        )

        all_points.append(radar_xyz)
        all_velocities.append(radar_vel_xy)
        all_offsets.extend([offset] * radar_xyz.shape[1])

    if not all_points:
        return []

    radar_xyz = np.hstack(all_points)
    radar_vel_xy = np.hstack(all_velocities)
    frame_offsets = np.asarray(all_offsets, dtype=float)

    objects = []

    for box_index, box in enumerate(boxes):
        if np.linalg.norm(box.center[:2]) > max_range:
            continue

        inside = points_in_box(box, radar_xyz)
        hit_count = int(np.sum(inside))

        if hit_count < min_hits:
            continue

        points_xy = radar_xyz[:2, inside].T
        velocities_xy = radar_vel_xy[:, inside].T
        offsets = frame_offsets[inside]
        centre_xy = box.center[:2].copy()

        velocity_estimate = compute_temporal_weighted_velocity(
            points_xy,
            velocities_xy,
            centre_xy,
            offsets,
            alpha=alpha,
        )

        objects.append({
            "box_index": box_index,
            "box_name": box.name,
            "position_xy": centre_xy,
            "velocity_xy": velocity_estimate,
            "radar_hits": hit_count,
            "associated_radar_points_xy": points_xy.tolist(),
        })

    return objects


def run_temporal_weighted_multiframe(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    min_hits: int = 1,
    max_range: float = 60.0,
    frame_window: int = 1,
    alpha: float = 0.7,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
) -> List[Dict]:
    """
    Run temporal-weighted multiframe fusion across a scene.
    """
    samples = get_scene_samples(nusc, scene_index=scene_index)
    frames = []

    for frame_index, sample in enumerate(samples):
        objects = extract_temporal_weighted_multiframe_measurements(
            nusc,
            samples,
            frame_index,
            radar_chan=radar_chan,
            ref_chan=ref_chan,
            min_hits=min_hits,
            max_range=max_range,
            frame_window=frame_window,
            alpha=alpha,
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_deg=yaw_deg,
        )

        frames.append({
            "frame_index": frame_index,
            "timestamp": sample["timestamp"],
            "objects": objects,
        })

    return frames


# --------------------------------------------------
# Motion compensation helper
# --------------------------------------------------

def motion_compensate_radar_points(
    radar_xyz: np.ndarray,
    radar_vel_xy: np.ndarray,
    dt_sec: float,
) -> np.ndarray:
    """
    Motion compensate radar points in the xy plane.

    This was tested as an alternative temporal recovery approach.
    It is kept here because it is a useful comparison method.
    """
    compensated = radar_xyz.copy()

    compensated[0, :] += radar_vel_xy[0, :] * dt_sec
    compensated[1, :] += radar_vel_xy[1, :] * dt_sec

    return compensated


def extract_motion_compensated_measurements(
    nusc,
    lidar_sample: dict,
    radar_sample: dict,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    min_hits: int = 1,
    max_range: float = 60.0,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
) -> List[Dict]:
    """
    Use radar from radar_sample, motion-compensated to lidar_sample time.

    This is not the main final method, but it was useful for testing whether
    simple velocity-based compensation helps.
    """
    boxes = get_lidar_boxes(nusc, lidar_sample, ref_chan=ref_chan)

    radar_xyz, radar_vel_xy = load_radar_xyz_velocity(
        nusc,
        radar_sample,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
        dx=dx,
        dy=dy,
        dz=dz,
        yaw_deg=yaw_deg,
    )

    dt_sec = (lidar_sample["timestamp"] - radar_sample["timestamp"]) / 1e6
    radar_xyz = motion_compensate_radar_points(
        radar_xyz,
        radar_vel_xy,
        dt_sec=dt_sec,
    )

    objects = []

    for box_index, box in enumerate(boxes):
        if np.linalg.norm(box.center[:2]) > max_range:
            continue

        inside = points_in_box(box, radar_xyz)
        hit_count = int(np.sum(inside))

        if hit_count < min_hits:
            continue

        points_xy = radar_xyz[:2, inside].T
        velocities_xy = radar_vel_xy[:, inside].T
        centre_xy = box.center[:2].copy()

        velocity_estimate = compute_weighted_velocity(
            points_xy,
            velocities_xy,
            centre_xy,
        )

        objects.append({
            "box_index": box_index,
            "box_name": box.name,
            "position_xy": centre_xy,
            "velocity_xy": velocity_estimate,
            "radar_hits": hit_count,
            "associated_radar_points_xy": points_xy.tolist(),
        })

    return objects


def run_scene_motion_compensated_temporal(
    nusc,
    scene_index: int = 0,
    frame_offset: int = -1,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    min_hits: int = 1,
    max_range: float = 60.0,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
) -> List[Dict]:
    """
    Run temporal frame-offset fusion with radar motion compensation.
    """
    samples = get_scene_samples(nusc, scene_index=scene_index)
    frames = []

    for frame_index, lidar_sample in enumerate(samples):
        radar_index = frame_index + frame_offset

        if radar_index < 0 or radar_index >= len(samples):
            continue

        radar_sample = samples[radar_index]

        objects = extract_motion_compensated_measurements(
            nusc,
            lidar_sample=lidar_sample,
            radar_sample=radar_sample,
            radar_chan=radar_chan,
            ref_chan=ref_chan,
            min_hits=min_hits,
            max_range=max_range,
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_deg=yaw_deg,
        )

        frames.append({
            "frame_index": frame_index,
            "radar_frame_index": radar_index,
            "timestamp": lidar_sample["timestamp"],
            "objects": objects,
        })

    return frames


# --------------------------------------------------
# Evaluation helpers
# --------------------------------------------------

def evaluate_multiframe_fusion(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    min_hits: int = 1,
    frame_offset: int = -1,
) -> Dict[str, Dict]:
    """
    Compare baseline, degraded, and naive multiframe fusion.
    """
    baseline = run_scene_fusion(
        nusc,
        scene_index=scene_index,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
        min_hits=min_hits,
    )

    degraded = run_scene_temporal_offset(
        nusc,
        scene_index=scene_index,
        frame_offset=frame_offset,
    )

    multiframe = run_scene_multiframe_fusion(
        nusc,
        scene_index=scene_index,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
        min_hits=min_hits,
    )

    return {
        "baseline": summarise_results(baseline),
        "degraded": summarise_results(degraded),
        "multiframe": summarise_results(multiframe),

        "baseline_frames": baseline,
        "degraded_frames": degraded,
        "multiframe_frames": multiframe,
    }


def evaluate_weighted_multiframe_fusion(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    min_hits: int = 1,
    frame_offset: int = -1,
    alpha: float = 0.7,
) -> Dict[str, Dict]:
    """
    Compare baseline, degraded, naive multiframe, and temporal-weighted multiframe.
    """
    baseline = run_scene_fusion(
        nusc,
        scene_index=scene_index,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
        min_hits=min_hits,
    )

    degraded = run_scene_temporal_offset(
        nusc,
        scene_index=scene_index,
        frame_offset=frame_offset,
    )

    multiframe = run_scene_multiframe_fusion(
        nusc,
        scene_index=scene_index,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
        min_hits=min_hits,
    )

    weighted_multiframe = run_temporal_weighted_multiframe(
        nusc,
        scene_index=scene_index,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
        min_hits=min_hits,
        alpha=alpha,
    )

    return {
        "baseline": summarise_results(baseline),
        "degraded": summarise_results(degraded),
        "multiframe": summarise_results(multiframe),
        "weighted_multiframe": summarise_results(weighted_multiframe),

        "baseline_frames": baseline,
        "degraded_frames": degraded,
        "multiframe_frames": multiframe,
        "weighted_multiframe_frames": weighted_multiframe,
    }


def multiframe_results_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Convert multiframe evaluation results into a compact table.
    """
    rows = []

    for name in ["baseline", "degraded", "multiframe", "weighted_multiframe"]:
        if name not in results:
            continue

        summary = results[name]

        rows.append({
            "Method": name.replace("_", " ").title(),
            "Objects per frame": summary.get("objects_per_frame", np.nan),
            "Total objects": summary.get("total_fused_objects", summary.get("total_objects", np.nan)),
            "Average speed": summary.get("avg_speed", np.nan),
        })

    return pd.DataFrame(rows)


def evaluate_motion_compensation_recovery(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    min_hits: int = 1,
    frame_offset: int = -1,
) -> Dict[str, Dict]:
    """
    Compare baseline, degraded, motion compensation, and multiframe fusion.

    This is useful for showing that motion compensation was tested but was
    not the strongest final method.
    """
    baseline = run_scene_fusion(
        nusc,
        scene_index=scene_index,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
        min_hits=min_hits,
    )

    degraded = run_scene_temporal_offset(
        nusc,
        scene_index=scene_index,
        frame_offset=frame_offset,
    )

    motion_comp = run_scene_motion_compensated_temporal(
        nusc,
        scene_index=scene_index,
        frame_offset=frame_offset,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
        min_hits=min_hits,
    )

    multiframe = run_scene_multiframe_fusion(
        nusc,
        scene_index=scene_index,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
        min_hits=min_hits,
    )

    return {
        "baseline": summarise_results(baseline),
        "degraded": summarise_results(degraded),
        "motion_comp": summarise_results(motion_comp),
        "multiframe": summarise_results(multiframe),

        "baseline_physical": compute_physical_alignment_metrics(baseline),
        "degraded_physical": compute_physical_alignment_metrics(degraded),
        "motion_comp_physical": compute_physical_alignment_metrics(motion_comp),
        "multiframe_physical": compute_physical_alignment_metrics(multiframe),

        "baseline_frames": baseline,
        "degraded_frames": degraded,
        "motion_comp_frames": motion_comp,
        "multiframe_frames": multiframe,
    }


# --------------------------------------------------
# Optional track-level comparison
# --------------------------------------------------

def evaluate_multiframe_track_level(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    min_hits: int = 1,
    frame_offset: int = -1,
    alpha: float = 0.7,
    gate_dist: float = 3.0,
    max_missed: int = 2,
) -> Dict[str, Dict]:
    """
    Run track-level comparison for baseline/degraded/multiframe methods.

    Tracking is imported inside this function to avoid making multiframe.py
    depend on tracking.py unless the user actually needs it.
    """
    from .tracking import (
        run_kalman_tracking_on_frames,
        summarise_tracks,
        summarise_tracked_frames,
    )

    eval_results = evaluate_weighted_multiframe_fusion(
        nusc,
        scene_index=scene_index,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
        min_hits=min_hits,
        frame_offset=frame_offset,
        alpha=alpha,
    )

    output = {}

    for key in [
        "baseline_frames",
        "degraded_frames",
        "multiframe_frames",
        "weighted_multiframe_frames",
    ]:
        method = key.replace("_frames", "")

        tracked_frames, tracks = run_kalman_tracking_on_frames(
            eval_results[key],
            gate_dist=gate_dist,
            max_missed=max_missed,
        )

        output[f"{method}_measurements"] = eval_results[method]
        output[f"{method}_tracks"] = summarise_tracks(tracks)
        output[f"{method}_track_states"] = summarise_tracked_frames(tracked_frames)
        output[f"{method}_frames"] = eval_results[key]

    return output


def track_level_multiframe_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build the track-level table used for multiframe comparison.
    """
    rows = []

    method_names = [
        ("baseline", "Baseline"),
        ("degraded", "Degraded"),
        ("multiframe", "Naive Multiframe"),
        ("weighted_multiframe", "Weighted Multiframe"),
    ]

    for key, label in method_names:
        meas = results.get(f"{key}_measurements", {})
        tracks = results.get(f"{key}_tracks", {})
        states = results.get(f"{key}_track_states", {})

        rows.append({
            "Method": label,
            "Meas count": meas.get("total_fused_objects", meas.get("total_objects", np.nan)),
            "Meas / frame": meas.get("objects_per_frame", np.nan),
            "Tracks": tracks.get("num_tracks", np.nan),
            "Track length": tracks.get("avg_track_length", np.nan),
            "Track states/frame": states.get("track_states_per_frame", np.nan),
            "Track average speed": tracks.get("avg_final_speed", np.nan),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("multiframe.py contains multiframe fusion and temporal aggregation methods.")