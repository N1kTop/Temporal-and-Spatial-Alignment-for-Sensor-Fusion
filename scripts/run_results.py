"""
run_results.py

Run the main project methods and print/save their result tables.

This is not just a test script. It is meant to produce the main numerical
outputs used to compare:

- baseline fusion
- temporal degradation
- interpolation recovery
- multiframe fusion
- temporal-weighted multiframe fusion
- motion compensation
- Kalman tracking
- constant-velocity vs constant-acceleration Kalman models
- spatial translation / yaw correction

Run from project root:

    python scripts/run_all_results.py --dataroot data/sets/nuscenes --version v1.0-mini

Outputs are saved in:

    results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import points_in_box


# Allow running from project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data_utils import (  # noqa: E402
    get_scene_samples,
    build_timestamp_table_from_scene_index,
    get_scene_sensor_channels,
)
from src.fusion import (  # noqa: E402
    run_scene_fusion,
    run_scene_temporal_offset,
    compute_weighted_velocity,
)
from src.multiframe import (  # noqa: E402
    run_scene_multiframe_fusion,
    run_temporal_weighted_multiframe,
    run_scene_motion_compensated_temporal,
)
from src.temporal_alignment import evaluate_pairing  # noqa: E402
from src.spatial_recovery import estimate_scene_correction_grid_physical  # noqa: E402
from src.tracking import (  # noqa: E402
    run_kalman_tracking_on_frames,
    summarise_tracks,
    compare_cv_and_ca_models,
)
from src.metrics import (  # noqa: E402
    summarise_results,
    fusion_summary_table,
    track_summary_table,
    save_multiple_tables_csv,
)


# --------------------------------------------------
# Display helpers
# --------------------------------------------------

def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_table(title: str, df: pd.DataFrame, decimals: int = 3) -> None:
    print_section(title)
    print(df.round(decimals).to_string(index=False))


def save_table(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    path = out_dir / filename
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


# --------------------------------------------------
# Interpolation recovery
# Kept here because the current src files do not yet have a separate
# interpolation module.
# --------------------------------------------------

def load_radar_for_sample(nusc, sample, radar_chan="RADAR_FRONT", ref_chan="LIDAR_TOP"):
    radar_pc, _ = RadarPointCloud.from_file_multisweep(
        nusc,
        sample,
        chan=radar_chan,
        ref_chan=ref_chan,
        nsweeps=1,
    )

    radar_xyz = radar_pc.points[:3, :]
    radar_vel_xy = np.vstack((radar_pc.points[8, :], radar_pc.points[9, :]))

    return radar_xyz, radar_vel_xy


def interpolate_velocity(v_prev, v_next, alpha):
    return (1.0 - alpha) * v_prev + alpha * v_next


def extract_interpolated_measurements(
    nusc,
    prev_sample,
    curr_sample,
    next_sample,
    radar_chan="RADAR_FRONT",
    ref_chan="LIDAR_TOP",
    min_hits=1,
    max_range=60.0,
):
    """
    Interpolation recovery method.

    Uses LiDAR boxes from the current frame.
    Uses radar from previous and next frames.
    Interpolates radar-derived velocity to the current timestamp.
    """
    lidar_sd = nusc.get("sample_data", curr_sample["data"][ref_chan])
    _, boxes, _ = nusc.get_sample_data(lidar_sd["token"])

    radar_prev, vel_prev = load_radar_for_sample(
        nusc,
        prev_sample,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
    )

    radar_next, vel_next = load_radar_for_sample(
        nusc,
        next_sample,
        radar_chan=radar_chan,
        ref_chan=ref_chan,
    )

    t_prev = prev_sample["timestamp"]
    t_curr = curr_sample["timestamp"]
    t_next = next_sample["timestamp"]

    alpha = 0.5 if t_next == t_prev else (t_curr - t_prev) / (t_next - t_prev)

    objects = []

    for box_index, box in enumerate(boxes):
        if np.linalg.norm(box.center[:2]) > max_range:
            continue

        mask_prev = points_in_box(box, radar_prev)
        mask_next = points_in_box(box, radar_next)

        hit_prev = int(np.sum(mask_prev))
        hit_next = int(np.sum(mask_next))

        if hit_prev < min_hits or hit_next < min_hits:
            continue

        centre_xy = box.center[:2].copy()

        v_prev = compute_weighted_velocity(
            radar_prev[:2, mask_prev].T,
            vel_prev[:, mask_prev].T,
            centre_xy,
        )

        v_next = compute_weighted_velocity(
            radar_next[:2, mask_next].T,
            vel_next[:, mask_next].T,
            centre_xy,
        )

        v_interp = interpolate_velocity(v_prev, v_next, alpha)

        associated_points = np.vstack([
            radar_prev[:2, mask_prev].T,
            radar_next[:2, mask_next].T,
        ])

        objects.append({
            "box_index": box_index,
            "box_name": box.name,
            "position_xy": centre_xy,
            "velocity_xy": v_interp,
            "radar_hits": hit_prev + hit_next,
            "associated_radar_points_xy": associated_points.tolist(),
        })

    return objects


def run_scene_interpolation(
    nusc,
    scene_index=0,
    radar_chan="RADAR_FRONT",
    ref_chan="LIDAR_TOP",
    min_hits=1,
    max_range=60.0,
):
    """
    Run interpolation recovery across a scene.

    First and last frames are skipped because interpolation needs neighbours.
    """
    samples = get_scene_samples(nusc, scene_index=scene_index)
    frames = []

    for i in range(1, len(samples) - 1):
        objects = extract_interpolated_measurements(
            nusc,
            prev_sample=samples[i - 1],
            curr_sample=samples[i],
            next_sample=samples[i + 1],
            radar_chan=radar_chan,
            ref_chan=ref_chan,
            min_hits=min_hits,
            max_range=max_range,
        )

        frames.append({
            "frame_index": i,
            "timestamp": samples[i]["timestamp"],
            "objects": objects,
        })

    return frames


# --------------------------------------------------
# Main result runner
# --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot", default="data/sets/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--radar-chan", default="RADAR_FRONT")
    parser.add_argument("--ref-chan", default="LIDAR_TOP")
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--out-dir", default="results")

    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print_section("Loading nuScenes")
    nusc = NuScenes(
        version=args.version,
        dataroot=args.dataroot,
        verbose=True,
    )

    print(f"Scene index: {args.scene_index}")
    print(f"Radar channel: {args.radar_chan}")
    print(f"Reference channel: {args.ref_chan}")
    print(f"Minimum radar hits: {args.min_hits}")

    # --------------------------------------------------
    # 1. Timestamp pairing results
    # --------------------------------------------------

    print_section("1. Timestamp pairing")

    timestamp_df = build_timestamp_table_from_scene_index(
        nusc,
        scene_index=args.scene_index,
    )

    sensors = get_scene_sensor_channels(nusc, scene_index=args.scene_index)
    sensors_to_eval = [s for s in sensors if s != args.ref_chan]

    timestamp_pairing_table, _ = evaluate_pairing(
        timestamp_df,
        ref_sensor=args.ref_chan,
        sensors=sensors_to_eval,
        threshold_ms=50.0,
    )

    print_table("Timestamp pairing summary", timestamp_pairing_table)
    save_table(timestamp_pairing_table, out_dir, "timestamp_pairing_summary.csv")

    # --------------------------------------------------
    # 2. Main fusion methods
    # --------------------------------------------------

    print_section("2. Running fusion methods")

    baseline_frames = run_scene_fusion(
        nusc,
        scene_index=args.scene_index,
        radar_chan=args.radar_chan,
        ref_chan=args.ref_chan,
        min_hits=args.min_hits,
    )

    degraded_minus_frames = run_scene_temporal_offset(
        nusc,
        scene_index=args.scene_index,
        frame_offset=-1,
    )

    degraded_plus_frames = run_scene_temporal_offset(
        nusc,
        scene_index=args.scene_index,
        frame_offset=1,
    )

    interpolation_frames = run_scene_interpolation(
        nusc,
        scene_index=args.scene_index,
        radar_chan=args.radar_chan,
        ref_chan=args.ref_chan,
        min_hits=args.min_hits,
    )

    multiframe_frames = run_scene_multiframe_fusion(
        nusc,
        scene_index=args.scene_index,
        radar_chan=args.radar_chan,
        ref_chan=args.ref_chan,
        min_hits=args.min_hits,
    )

    weighted_multiframe_frames = run_temporal_weighted_multiframe(
        nusc,
        scene_index=args.scene_index,
        radar_chan=args.radar_chan,
        ref_chan=args.ref_chan,
        min_hits=args.min_hits,
    )

    motion_comp_frames = run_scene_motion_compensated_temporal(
        nusc,
        scene_index=args.scene_index,
        frame_offset=-1,
        radar_chan=args.radar_chan,
        ref_chan=args.ref_chan,
        min_hits=args.min_hits,
    )

    fusion_methods = {
        "Baseline": baseline_frames,
        "Degraded (-1)": degraded_minus_frames,
        "Degraded (+1)": degraded_plus_frames,
        "Interpolation": interpolation_frames,
        "Multiframe": multiframe_frames,
        "Weighted Multiframe": weighted_multiframe_frames,
        "Motion Compensation": motion_comp_frames,
    }

    fusion_table = fusion_summary_table(fusion_methods)

    print_table("Fusion method comparison", fusion_table)
    save_table(fusion_table, out_dir, "fusion_method_comparison.csv")

    # --------------------------------------------------
    # 3. Temporal offset sweep
    # --------------------------------------------------

    print_section("3. Temporal offset sweep")

    temporal_offset_rows = []

    for offset in [0, -1, 1, -2, 2]:
        if offset == 0:
            frames = baseline_frames
        else:
            frames = run_scene_temporal_offset(
                nusc,
                scene_index=args.scene_index,
                frame_offset=offset,
            )

        summary = summarise_results(frames)

        temporal_offset_rows.append({
            "frame_offset": offset,
            "total_fused_objects": summary["total_fused_objects"],
            "objects_per_frame": summary["objects_per_frame"],
            "avg_hits": summary["avg_hits"],
            "avg_speed_mps": summary["avg_speed"],
        })

    temporal_offset_table = pd.DataFrame(temporal_offset_rows)

    print_table("Temporal offset sweep", temporal_offset_table)
    save_table(temporal_offset_table, out_dir, "temporal_offset_sweep.csv")

    # --------------------------------------------------
    # 4. Kalman tracking results
    # --------------------------------------------------

    print_section("4. Kalman tracking")

    track_summaries = {}

    tracked_frame_outputs = {}

    for method_name, frames in {
        "Baseline": baseline_frames,
        "Degraded (-1)": degraded_minus_frames,
        "Interpolation": interpolation_frames,
        "Multiframe": multiframe_frames,
        "Weighted Multiframe": weighted_multiframe_frames,
    }.items():
        tracked_frames, tracks = run_kalman_tracking_on_frames(frames)
        tracked_frame_outputs[method_name] = tracked_frames
        track_summaries[method_name] = summarise_tracks(tracks)

    tracking_table = track_summary_table(track_summaries)

    print_table("Kalman tracking comparison", tracking_table)
    save_table(tracking_table, out_dir, "kalman_tracking_comparison.csv")

    # --------------------------------------------------
    # 5. Constant velocity vs acceleration
    # --------------------------------------------------

    print_section("5. Constant velocity vs constant acceleration")

    cv_ca_table = compare_cv_and_ca_models(
        degraded_minus_frames,
        gate_dist=3.0,
        max_missed=2,
    )

    print_table("Constant velocity vs acceleration", cv_ca_table)
    save_table(cv_ca_table, out_dir, "cv_vs_ca_kalman.csv")

    # --------------------------------------------------
    # 6. Spatial correction: translation
    # --------------------------------------------------

    print_section("6. Spatial correction: translation")

    dx_candidates = np.linspace(-1.5, 0.5, 9)

    best_dx, all_dx = estimate_scene_correction_grid_physical(
        nusc,
        scene_index=args.scene_index,
        radar_chan=args.radar_chan,
        ref_chan=args.ref_chan,
        degrade_dx=1.0,
        degrade_dy=0.0,
        degrade_yaw_deg=0.0,
        dx_candidates=dx_candidates,
        dy_candidates=[0.0],
        yaw_candidates=[0.0],
        max_range=60.0,
    )

    dx_table = pd.DataFrame(all_dx)

    print("Best translation correction:")
    print(best_dx)
    print_table("Translation correction candidates", dx_table)
    save_table(dx_table, out_dir, "translation_correction_candidates.csv")

    # --------------------------------------------------
    # 7. Spatial correction: yaw
    # --------------------------------------------------

    print_section("7. Spatial correction: yaw")

    yaw_candidates = [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0]

    best_yaw, all_yaw = estimate_scene_correction_grid_physical(
        nusc,
        scene_index=args.scene_index,
        radar_chan=args.radar_chan,
        ref_chan=args.ref_chan,
        degrade_dx=0.0,
        degrade_dy=0.0,
        degrade_yaw_deg=3.0,
        dx_candidates=[0.0],
        dy_candidates=[0.0],
        yaw_candidates=yaw_candidates,
        max_range=60.0,
    )

    yaw_table = pd.DataFrame(all_yaw)

    print("Best yaw correction:")
    print(best_yaw)
    print_table("Yaw correction candidates", yaw_table)
    save_table(yaw_table, out_dir, "yaw_correction_candidates.csv")

    # --------------------------------------------------
    # 8. Combined CSV
    # --------------------------------------------------

    print_section("8. Saving combined results CSV")

    all_tables = {
        "Timestamp Pairing Summary": timestamp_pairing_table,
        "Fusion Method Comparison": fusion_table,
        "Temporal Offset Sweep": temporal_offset_table,
        "Kalman Tracking Comparison": tracking_table,
        "CV vs CA Kalman": cv_ca_table,
        "Translation Correction Candidates": dx_table,
        "Yaw Correction Candidates": yaw_table,
    }

    combined_path = out_dir / "all_results_tables.csv"
    save_multiple_tables_csv(all_tables, combined_path)

    print_section("Finished")
    print(f"Result tables saved in: {out_dir}")


if __name__ == "__main__":
    main()