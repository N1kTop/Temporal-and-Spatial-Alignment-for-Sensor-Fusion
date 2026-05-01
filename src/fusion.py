"""
fusion.py

Core LiDAR–radar fusion logic.

This file handles:
- associating radar points to LiDAR boxes
- estimating object velocities from radar
- running fusion across frames/scenes

This is intentionally written in a simple and readable way.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import points_in_box

from .data_utils import get_scene_samples


# --------------------------------------------------
# Core velocity estimation
# --------------------------------------------------

def compute_weighted_velocity(points_xy: np.ndarray,
                              radar_vel_xy: np.ndarray,
                              box_center_xy: np.ndarray,
                              eps: float = 1e-6) -> np.ndarray:
    """
    Estimate object velocity from radar points using distance weighting.

    Points closer to the box centre have higher influence.
    """
    if len(points_xy) == 0:
        return np.array([0.0, 0.0])

    dists = np.linalg.norm(points_xy - box_center_xy[None, :], axis=1)
    weights = 1.0 / (dists + eps)

    if np.sum(weights) <= 0:
        return np.mean(radar_vel_xy, axis=0)

    v = np.sum(radar_vel_xy * weights[:, None], axis=0) / np.sum(weights)
    return v


# --------------------------------------------------
# Radar extraction
# --------------------------------------------------

def load_radar_points(nusc, sample: dict,
                      radar_chan: str = "RADAR_FRONT",
                      ref_chan: str = "LIDAR_TOP"):
    """
    Load radar point cloud and return:
        xyz (3, N)
        velocity_xy (2, N)
    """
    radar_pc, _ = RadarPointCloud.from_file_multisweep(
        nusc,
        sample,
        chan=radar_chan,
        ref_chan=ref_chan,
        nsweeps=1
    )

    xyz = radar_pc.points[:3, :]
    vel_xy = np.vstack((radar_pc.points[8, :], radar_pc.points[9, :]))

    return xyz, vel_xy


# --------------------------------------------------
# Single-frame fusion
# --------------------------------------------------

def extract_fused_objects(nusc,
                          sample: dict,
                          radar_chan: str = "RADAR_FRONT",
                          ref_chan: str = "LIDAR_TOP",
                          min_hits: int = 1,
                          max_range: float = 60.0) -> List[Dict]:
    """
    Associate radar points to LiDAR boxes for a single frame.

    Returns a list of fused objects with:
        position
        velocity
        radar hit count
    """
    lidar_sd = nusc.get("sample_data", sample["data"][ref_chan])
    _, boxes, _ = nusc.get_sample_data(lidar_sd["token"])

    radar_xyz, radar_vel_xy = load_radar_points(
        nusc,
        sample,
        radar_chan=radar_chan,
        ref_chan=ref_chan
    )

    objects = []

    for box_index, box in enumerate(boxes):

        # skip far objects
        if np.linalg.norm(box.center[:2]) > max_range:
            continue

        inside = points_in_box(box, radar_xyz)
        hit_count = int(np.sum(inside))

        if hit_count < min_hits:
            continue

        points_xy = radar_xyz[:2, inside].T
        vel_xy = radar_vel_xy[:, inside].T
        centre_xy = box.center[:2].copy()

        v_est = compute_weighted_velocity(points_xy, vel_xy, centre_xy)

        objects.append({
            "box_index": box_index,
            "box_name": box.name,
            "position_xy": centre_xy,
            "velocity_xy": v_est,
            "radar_hits": hit_count,
            "associated_radar_points_xy": points_xy.tolist(),
        })

    return objects


# --------------------------------------------------
# Scene-level fusion
# --------------------------------------------------

def run_scene_fusion(nusc,
                     scene_index: int = 0,
                     radar_chan: str = "RADAR_FRONT",
                     ref_chan: str = "LIDAR_TOP",
                     min_hits: int = 1) -> List[Dict]:
    """
    Run baseline fusion across an entire scene.

    Returns list of frames:
        {
            frame_index,
            timestamp,
            objects
        }
    """
    samples = get_scene_samples(nusc, scene_index=scene_index)

    frames = []

    for i, sample in enumerate(samples):
        objects = extract_fused_objects(
            nusc,
            sample,
            radar_chan=radar_chan,
            ref_chan=ref_chan,
            min_hits=min_hits
        )

        frames.append({
            "frame_index": i,
            "timestamp": sample["timestamp"],
            "objects": objects
        })

    return frames


# --------------------------------------------------
# Temporal offset fusion (used in experiments)
# --------------------------------------------------

def extract_temporally_offset_objects(nusc,
                                       lidar_sample: dict,
                                       radar_sample: dict,
                                       radar_chan: str = "RADAR_FRONT",
                                       ref_chan: str = "LIDAR_TOP",
                                       min_hits: int = 1,
                                       max_range: float = 60.0):
    """
    Fuse LiDAR boxes from one frame with radar from another frame.

    This is used to simulate temporal misalignment.
    """
    lidar_sd = nusc.get("sample_data", lidar_sample["data"][ref_chan])
    _, boxes, _ = nusc.get_sample_data(lidar_sd["token"])

    radar_xyz, radar_vel_xy = load_radar_points(
        nusc,
        radar_sample,
        radar_chan=radar_chan,
        ref_chan=ref_chan
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
        vel_xy = radar_vel_xy[:, inside].T
        centre_xy = box.center[:2].copy()

        v_est = compute_weighted_velocity(points_xy, vel_xy, centre_xy)

        objects.append({
            "box_index": box_index,
            "box_name": box.name,
            "position_xy": centre_xy,
            "velocity_xy": v_est,
            "radar_hits": hit_count,
        })

    return objects


def run_scene_temporal_offset(nusc,
                              scene_index: int = 0,
                              frame_offset: int = 1):
    """
    Run fusion using temporally shifted radar frames.

    frame_offset:
        0 = baseline
        +1 = future radar
        -1 = past radar
    """
    samples = get_scene_samples(nusc, scene_index=scene_index)

    frames = []

    for i, lidar_sample in enumerate(samples):
        j = i + frame_offset

        if j < 0 or j >= len(samples):
            continue

        radar_sample = samples[j]

        objects = extract_temporally_offset_objects(
            nusc,
            lidar_sample,
            radar_sample
        )

        frames.append({
            "frame_index": i,
            "timestamp": lidar_sample["timestamp"],
            "objects": objects
        })

    return frames


# --------------------------------------------------
# Simple summary helpers
# --------------------------------------------------

def summarise_fusion_results(frames: List[Dict]) -> Dict:
    """
    Compute basic statistics for fusion output.
    """
    hits = []
    speeds = []

    for frame in frames:
        for obj in frame["objects"]:
            hits.append(obj["radar_hits"])

            vx, vy = obj["velocity_xy"]
            speeds.append(np.sqrt(vx**2 + vy**2))

    total_frames = len(frames)
    total_objects = sum(len(f["objects"]) for f in frames)

    return {
        "total_frames": total_frames,
        "total_objects": total_objects,
        "objects_per_frame": total_objects / total_frames if total_frames else 0.0,
        "avg_hits": float(np.mean(hits)) if hits else 0.0,
        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
    }


if __name__ == "__main__":
    print("fusion.py contains LiDAR–radar fusion logic.")