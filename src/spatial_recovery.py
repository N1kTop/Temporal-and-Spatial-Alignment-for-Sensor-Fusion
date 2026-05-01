"""
spatial_recovery.py

Scene-level spatial alignment recovery for LiDAR-radar fusion.

This file contains the spatial correction logic used in the project:
- overlap-based scoring
- translation / yaw grid search
- correction after known degradation
- physical distance penalty scoring

The idea is simple:
    apply a candidate correction to radar points,
    check how well they agree with LiDAR boxes,
    choose the correction with the best scene-level score.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import points_in_box

from .data_utils import get_scene_samples
from .degradation import apply_spatial_transform


# --------------------------------------------------
# Basic scoring helpers
# --------------------------------------------------

def _load_radar_and_boxes(
    nusc,
    sample: dict,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
):
    """
    Load radar points and LiDAR boxes for one sample.

    Radar points are returned in the reference sensor frame.
    """
    lidar_sd = nusc.get("sample_data", sample["data"][ref_chan])

    radar_pc, _ = RadarPointCloud.from_file_multisweep(
        nusc,
        sample,
        chan=radar_chan,
        ref_chan=ref_chan,
        nsweeps=1,
    )

    radar_xyz = radar_pc.points[:3, :].copy()
    _, boxes, _ = nusc.get_sample_data(lidar_sd["token"])

    return radar_xyz, boxes


def score_radar_box_overlap(
    radar_xyz: np.ndarray,
    boxes: Sequence,
    max_range: float = 60.0,
) -> Dict[str, float]:
    """
    Count how many radar points fall inside LiDAR boxes.

    This is the simple overlap-based score used first in the project.
    """
    total_hits = 0
    matched_boxes = 0

    for box in boxes:
        if np.linalg.norm(box.center[:2]) > max_range:
            continue

        inside_box = points_in_box(box, radar_xyz)
        hit_count = int(np.sum(inside_box))

        total_hits += hit_count

        if hit_count > 0:
            matched_boxes += 1

    return {
        "matched_boxes": matched_boxes,
        "total_hits": total_hits,
    }


def make_overlap_score(
    matched_boxes: int,
    total_hits: int,
    w_boxes: float = 1000.0,
    w_hits: float = 1.0,
) -> float:
    """
    Convert overlap statistics into one scalar score.

    Matched boxes are weighted much more strongly than raw hit count.
    """
    return float(w_boxes * matched_boxes + w_hits * total_hits)


# --------------------------------------------------
# Single-sample spatial scoring
# --------------------------------------------------

def compute_association_score(
    nusc,
    sample: dict,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
    max_range: float = 60.0,
) -> Dict[str, float]:
    """
    Score one sample after applying a candidate spatial transform.

    Higher matched_boxes / total_hits means stronger LiDAR-radar agreement.
    """
    radar_xyz, boxes = _load_radar_and_boxes(
        nusc,
        sample,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
    )

    radar_xyz = apply_spatial_transform(
        radar_xyz,
        dx=dx,
        dy=dy,
        dz=dz,
        yaw_deg=yaw_deg,
    )

    return score_radar_box_overlap(
        radar_xyz,
        boxes,
        max_range=max_range,
    )


def compute_association_score_with_degradation(
    nusc,
    sample: dict,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    degrade_dx: float = 0.0,
    degrade_dy: float = 0.0,
    degrade_dz: float = 0.0,
    degrade_yaw_deg: float = 0.0,
    correct_dx: float = 0.0,
    correct_dy: float = 0.0,
    correct_dz: float = 0.0,
    correct_yaw_deg: float = 0.0,
    max_range: float = 60.0,
) -> Dict[str, float]:
    """
    Apply a known degradation, then apply a candidate correction, then score.

    This mirrors the validation logic used in the notebook:
        radar -> degraded radar -> corrected radar -> score
    """
    radar_xyz, boxes = _load_radar_and_boxes(
        nusc,
        sample,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
    )

    radar_xyz = apply_spatial_transform(
        radar_xyz,
        dx=degrade_dx,
        dy=degrade_dy,
        dz=degrade_dz,
        yaw_deg=degrade_yaw_deg,
    )

    radar_xyz = apply_spatial_transform(
        radar_xyz,
        dx=correct_dx,
        dy=correct_dy,
        dz=correct_dz,
        yaw_deg=correct_yaw_deg,
    )

    return score_radar_box_overlap(
        radar_xyz,
        boxes,
        max_range=max_range,
    )


def estimate_correction_grid_for_sample(
    nusc,
    sample: dict,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    degrade_dx: float = 0.0,
    degrade_dy: float = 0.0,
    degrade_dz: float = 0.0,
    degrade_yaw_deg: float = 0.0,
    dx_candidates: Optional[Sequence[float]] = None,
    dy_candidates: Optional[Sequence[float]] = None,
    yaw_candidates: Optional[Sequence[float]] = None,
    max_range: float = 60.0,
) -> Dict[str, float]:
    """
    Brute-force grid search for one sample.

    This is useful for quick sanity checks, but scene-level scoring is usually
    more reliable.
    """
    dx_candidates = [0.0] if dx_candidates is None else dx_candidates
    dy_candidates = [0.0] if dy_candidates is None else dy_candidates
    yaw_candidates = [0.0] if yaw_candidates is None else yaw_candidates

    best_result = None
    best_score = -np.inf

    for dx in dx_candidates:
        for dy in dy_candidates:
            for yaw in yaw_candidates:
                stats = compute_association_score_with_degradation(
                    nusc,
                    sample,
                    radar_chan=radar_chan,
                    ref_chan=ref_chan,
                    degrade_dx=degrade_dx,
                    degrade_dy=degrade_dy,
                    degrade_dz=degrade_dz,
                    degrade_yaw_deg=degrade_yaw_deg,
                    correct_dx=float(dx),
                    correct_dy=float(dy),
                    correct_yaw_deg=float(yaw),
                    max_range=max_range,
                )

                score = make_overlap_score(
                    stats["matched_boxes"],
                    stats["total_hits"],
                )

                result = {
                    "estimated_dx": float(dx),
                    "estimated_dy": float(dy),
                    "estimated_yaw_deg": float(yaw),
                    "matched_boxes": stats["matched_boxes"],
                    "total_hits": stats["total_hits"],
                    "score_value": score,
                }

                if score > best_score:
                    best_score = score
                    best_result = result

    return best_result


# --------------------------------------------------
# Scene-level overlap scoring
# --------------------------------------------------

def compute_scene_association_score_with_degradation(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    degrade_dx: float = 0.0,
    degrade_dy: float = 0.0,
    degrade_dz: float = 0.0,
    degrade_yaw_deg: float = 0.0,
    correct_dx: float = 0.0,
    correct_dy: float = 0.0,
    correct_dz: float = 0.0,
    correct_yaw_deg: float = 0.0,
    max_range: float = 60.0,
) -> Dict[str, float]:
    """
    Compute overlap score over a full scene.

    Scene-level scoring is less noisy than single-frame scoring.
    """
    samples = get_scene_samples(nusc, scene_index=scene_index)

    total_hits_scene = 0
    matched_boxes_scene = 0
    frames_used = 0

    for sample in samples:
        stats = compute_association_score_with_degradation(
            nusc,
            sample,
            radar_chan=radar_chan,
            ref_chan=ref_chan,
            degrade_dx=degrade_dx,
            degrade_dy=degrade_dy,
            degrade_dz=degrade_dz,
            degrade_yaw_deg=degrade_yaw_deg,
            correct_dx=correct_dx,
            correct_dy=correct_dy,
            correct_dz=correct_dz,
            correct_yaw_deg=correct_yaw_deg,
            max_range=max_range,
        )

        total_hits_scene += stats["total_hits"]
        matched_boxes_scene += stats["matched_boxes"]
        frames_used += 1

    return {
        "frames_used": frames_used,
        "matched_boxes_scene": matched_boxes_scene,
        "total_hits_scene": total_hits_scene,
    }


def estimate_scene_correction_grid(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    degrade_dx: float = 0.0,
    degrade_dy: float = 0.0,
    degrade_dz: float = 0.0,
    degrade_yaw_deg: float = 0.0,
    dx_candidates: Optional[Sequence[float]] = None,
    dy_candidates: Optional[Sequence[float]] = None,
    yaw_candidates: Optional[Sequence[float]] = None,
    max_range: float = 60.0,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Estimate correction by maximising scene-level overlap score.
    """
    dx_candidates = [0.0] if dx_candidates is None else dx_candidates
    dy_candidates = [0.0] if dy_candidates is None else dy_candidates
    yaw_candidates = [0.0] if yaw_candidates is None else yaw_candidates

    best_result = None
    best_score = -np.inf
    all_results = []

    for dx in dx_candidates:
        for dy in dy_candidates:
            for yaw in yaw_candidates:
                stats = compute_scene_association_score_with_degradation(
                    nusc,
                    scene_index=scene_index,
                    radar_chan=radar_chan,
                    ref_chan=ref_chan,
                    degrade_dx=degrade_dx,
                    degrade_dy=degrade_dy,
                    degrade_dz=degrade_dz,
                    degrade_yaw_deg=degrade_yaw_deg,
                    correct_dx=float(dx),
                    correct_dy=float(dy),
                    correct_yaw_deg=float(yaw),
                    max_range=max_range,
                )

                score = make_overlap_score(
                    stats["matched_boxes_scene"],
                    stats["total_hits_scene"],
                )

                result = {
                    "estimated_dx": float(dx),
                    "estimated_dy": float(dy),
                    "estimated_yaw_deg": float(yaw),
                    "matched_boxes_scene": stats["matched_boxes_scene"],
                    "total_hits_scene": stats["total_hits_scene"],
                    "frames_used": stats["frames_used"],
                    "score_value": score,
                }

                all_results.append(result)

                if score > best_score:
                    best_score = score
                    best_result = result

    return best_result, all_results


# --------------------------------------------------
# Physical distance scoring
# --------------------------------------------------

def compute_scene_score_with_distance_penalty(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    degrade_dx: float = 0.0,
    degrade_dy: float = 0.0,
    degrade_dz: float = 0.0,
    degrade_yaw_deg: float = 0.0,
    correct_dx: float = 0.0,
    correct_dy: float = 0.0,
    correct_dz: float = 0.0,
    correct_yaw_deg: float = 0.0,
    max_range: float = 60.0,
) -> Dict[str, float]:
    """
    Compute scene-level alignment stats with a physical distance metric.

    The centre distance measures how far associated radar points are from
    the LiDAR box centre. Lower is generally better.
    """
    samples = get_scene_samples(nusc, scene_index=scene_index)

    frames_used = 0
    matched_boxes_scene = 0
    total_hits_scene = 0
    centre_distances = []

    for sample in samples:
        radar_xyz, boxes = _load_radar_and_boxes(
            nusc,
            sample,
            radar_chan=radar_chan,
            ref_chan=ref_chan,
        )

        radar_xyz = apply_spatial_transform(
            radar_xyz,
            dx=degrade_dx,
            dy=degrade_dy,
            dz=degrade_dz,
            yaw_deg=degrade_yaw_deg,
        )

        radar_xyz = apply_spatial_transform(
            radar_xyz,
            dx=correct_dx,
            dy=correct_dy,
            dz=correct_dz,
            yaw_deg=correct_yaw_deg,
        )

        for box in boxes:
            if np.linalg.norm(box.center[:2]) > max_range:
                continue

            inside_box = points_in_box(box, radar_xyz)
            hit_count = int(np.sum(inside_box))

            total_hits_scene += hit_count

            if hit_count > 0:
                matched_boxes_scene += 1

                points_xy = radar_xyz[:2, inside_box].T
                dists = np.linalg.norm(points_xy - box.center[:2], axis=1)
                centre_distances.extend(dists.tolist())

        frames_used += 1

    mean_centre_distance = (
        float(np.mean(centre_distances)) if centre_distances else np.inf
    )

    median_centre_distance = (
        float(np.median(centre_distances)) if centre_distances else np.inf
    )

    return {
        "frames_used": frames_used,
        "matched_boxes_scene": matched_boxes_scene,
        "total_hits_scene": total_hits_scene,
        "mean_centre_distance": mean_centre_distance,
        "median_centre_distance": median_centre_distance,
    }


def make_physical_score(
    matched_boxes: int,
    total_hits: int,
    mean_centre_distance: float,
    w_boxes: float = 1000.0,
    w_hits: float = 10.0,
    w_dist: float = 50.0,
) -> float:
    """
    Physically aware score used for alignment correction.

    score = w_boxes * matched_boxes
          + w_hits  * total_hits
          - w_dist  * mean_centre_distance
    """
    if not np.isfinite(mean_centre_distance):
        mean_centre_distance = 1e6

    return float(
        w_boxes * matched_boxes
        + w_hits * total_hits
        - w_dist * mean_centre_distance
    )


def estimate_scene_correction_grid_physical(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    degrade_dx: float = 0.0,
    degrade_dy: float = 0.0,
    degrade_dz: float = 0.0,
    degrade_yaw_deg: float = 0.0,
    dx_candidates: Optional[Sequence[float]] = None,
    dy_candidates: Optional[Sequence[float]] = None,
    yaw_candidates: Optional[Sequence[float]] = None,
    max_range: float = 60.0,
    w_boxes: float = 1000.0,
    w_hits: float = 10.0,
    w_dist: float = 50.0,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Estimate correction using scene-level overlap + centre-distance penalty.

    This is the more reliable correction method used for the final report.
    """
    dx_candidates = [0.0] if dx_candidates is None else dx_candidates
    dy_candidates = [0.0] if dy_candidates is None else dy_candidates
    yaw_candidates = [0.0] if yaw_candidates is None else yaw_candidates

    best_result = None
    best_score = -np.inf
    all_results = []

    for dx in dx_candidates:
        for dy in dy_candidates:
            for yaw in yaw_candidates:
                stats = compute_scene_score_with_distance_penalty(
                    nusc,
                    scene_index=scene_index,
                    radar_chan=radar_chan,
                    ref_chan=ref_chan,
                    degrade_dx=degrade_dx,
                    degrade_dy=degrade_dy,
                    degrade_dz=degrade_dz,
                    degrade_yaw_deg=degrade_yaw_deg,
                    correct_dx=float(dx),
                    correct_dy=float(dy),
                    correct_yaw_deg=float(yaw),
                    max_range=max_range,
                )

                score = make_physical_score(
                    matched_boxes=stats["matched_boxes_scene"],
                    total_hits=stats["total_hits_scene"],
                    mean_centre_distance=stats["mean_centre_distance"],
                    w_boxes=w_boxes,
                    w_hits=w_hits,
                    w_dist=w_dist,
                )

                result = {
                    "estimated_dx": float(dx),
                    "estimated_dy": float(dy),
                    "estimated_yaw_deg": float(yaw),
                    "matched_boxes_scene": stats["matched_boxes_scene"],
                    "total_hits_scene": stats["total_hits_scene"],
                    "mean_centre_distance": stats["mean_centre_distance"],
                    "median_centre_distance": stats["median_centre_distance"],
                    "frames_used": stats["frames_used"],
                    "score_value": score,
                }

                all_results.append(result)

                if score > best_score:
                    best_score = score
                    best_result = result

    return best_result, all_results


# --------------------------------------------------
# Physical alignment summaries
# --------------------------------------------------

def run_scene_with_transform_and_points(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    min_hits: int = 1,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    yaw_deg: float = 0.0,
    max_range: float = 60.0,
) -> List[Dict]:
    """
    Run object-level association while storing associated radar points.

    This is mainly used to compute physical metrics such as centre distance.
    """
    samples = get_scene_samples(nusc, scene_index=scene_index)
    frames = []

    for frame_index, sample in enumerate(samples):
        radar_xyz, boxes = _load_radar_and_boxes(
            nusc,
            sample,
            radar_chan=radar_chan,
            ref_chan=ref_chan,
        )

        radar_pc, _ = RadarPointCloud.from_file_multisweep(
            nusc,
            sample,
            chan=radar_chan,
            ref_chan=ref_chan,
            nsweeps=1,
        )

        radar_vel = np.vstack((radar_pc.points[8, :], radar_pc.points[9, :]))

        radar_xyz = apply_spatial_transform(
            radar_xyz,
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_deg=yaw_deg,
        )

        objects = []

        for box_index, box in enumerate(boxes):
            if np.linalg.norm(box.center[:2]) > max_range:
                continue

            inside_box = points_in_box(box, radar_xyz)
            hit_count = int(np.sum(inside_box))

            if hit_count < min_hits:
                continue

            assoc_pts_xy = radar_xyz[:2, inside_box].T
            assoc_vel_xy = radar_vel[:, inside_box].T

            # Median velocity is enough here. The important output is the
            # associated point cloud for physical distance checks.
            v_est = np.median(assoc_vel_xy, axis=0)

            objects.append({
                "box_index": box_index,
                "box_name": box.name,
                "position_xy": box.center[:2].copy(),
                "velocity_xy": v_est,
                "radar_hits": hit_count,
                "associated_radar_points_xy": assoc_pts_xy.tolist(),
            })

        frames.append({
            "frame_index": frame_index,
            "timestamp": sample["timestamp"],
            "objects": objects,
        })

    return frames


def compute_physical_alignment_metrics(frames: Sequence[Dict]) -> Dict[str, float]:
    """
    Compute simple physical metrics from frames containing associated radar points.
    """
    centre_distances = []
    points_per_object = []

    for frame in frames:
        for obj in frame["objects"]:
            box_center = np.asarray(obj["position_xy"], dtype=float)
            radar_points_xy = np.asarray(
                obj.get("associated_radar_points_xy", []),
                dtype=float,
            )

            if radar_points_xy.size == 0:
                continue

            radar_points_xy = radar_points_xy.reshape(-1, 2)
            dists = np.linalg.norm(radar_points_xy - box_center[None, :], axis=1)

            centre_distances.extend(dists.tolist())
            points_per_object.append(len(radar_points_xy))

    return {
        "mean_centre_distance": float(np.mean(centre_distances)) if centre_distances else np.nan,
        "median_centre_distance": float(np.median(centre_distances)) if centre_distances else np.nan,
        "mean_points_per_object": float(np.mean(points_per_object)) if points_per_object else np.nan,
        "median_points_per_object": float(np.median(points_per_object)) if points_per_object else np.nan,
    }


def compare_physical_cases(
    nusc,
    scene_index: int = 0,
    radar_chan: str = "RADAR_FRONT",
    ref_chan: str = "LIDAR_TOP",
    degrade_dx: float = 0.0,
    degrade_yaw_deg: float = 0.0,
    estimated_dx: float = 0.0,
    estimated_yaw_deg: float = 0.0,
    true_dx: float = 0.0,
    true_yaw_deg: float = 0.0,
) -> pd.DataFrame:
    """
    Build the physical comparison table used in the report.

    It compares:
        baseline
        degraded
        estimated correction
        true correction
    """
    cases = {
        "Baseline": {
            "dx": 0.0,
            "yaw_deg": 0.0,
        },
        "Degraded": {
            "dx": degrade_dx,
            "yaw_deg": degrade_yaw_deg,
        },
        "Estimated Correction": {
            "dx": degrade_dx + estimated_dx,
            "yaw_deg": degrade_yaw_deg + estimated_yaw_deg,
        },
        "True Correction": {
            "dx": degrade_dx + true_dx,
            "yaw_deg": degrade_yaw_deg + true_yaw_deg,
        },
    }

    rows = []

    for name, params in cases.items():
        frames = run_scene_with_transform_and_points(
            nusc,
            scene_index=scene_index,
            radar_chan=radar_chan,
            ref_chan=ref_chan,
            dx=params["dx"],
            yaw_deg=params["yaw_deg"],
        )

        metrics = compute_physical_alignment_metrics(frames)

        rows.append({
            "case": name,
            **metrics,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Display helpers
# --------------------------------------------------

def results_to_dataframe(results: Sequence[Dict]) -> pd.DataFrame:
    """
    Convert grid-search results into a dataframe.
    """
    return pd.DataFrame(list(results))


def print_sorted_results(
    results: Sequence[Dict],
    sort_key: str = "score_value",
    descending: bool = True,
    max_rows: Optional[int] = None,
) -> None:
    """
    Print grid-search candidates sorted by score.

    Useful in notebooks when checking why a correction was selected.
    """
    sorted_results = sorted(
        results,
        key=lambda r: r.get(sort_key, -np.inf),
        reverse=descending,
    )

    if max_rows is not None:
        sorted_results = sorted_results[:max_rows]

    for result in sorted_results:
        print(result)


if __name__ == "__main__":
    print("spatial_recovery.py contains spatial correction and scene-level scoring methods.")