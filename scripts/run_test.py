"""
run_test.py

Quick project-level test script.

This checks that the main modules work together:
- baseline fusion
- temporal degradation
- multiframe fusion
- weighted multiframe fusion
- Kalman tracking
- spatial correction scoring
- summary metrics

It is not meant to reproduce every report figure.
It is just a practical sanity check for the GitHub code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes


# Allow running from project root without installing as a package.
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
)
from src.multiframe import (  # noqa: E402
    run_scene_multiframe_fusion,
    run_temporal_weighted_multiframe,
)
from src.tracking import (  # noqa: E402
    run_kalman_tracking_on_frames,
    summarise_tracks,
    compare_cv_and_ca_models,
)
from src.metrics import (  # noqa: E402
    summarise_results,
    fusion_summary_table,
    track_summary_table,
)
from src.temporal_alignment import (  # noqa: E402
    evaluate_pairing,
)
from src.spatial_recovery import (  # noqa: E402
    estimate_scene_correction_grid_physical,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="data/sets/nuscenes")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--radar-chan", default="RADAR_FRONT")
    parser.add_argument("--ref-chan", default="LIDAR_TOP")
    parser.add_argument("--min-hits", type=int, default=1)
    args = parser.parse_args()

    print_section("Loading nuScenes")
    nusc = NuScenes(
        version=args.version,
        dataroot=args.dataroot,
        verbose=True,
    )

    samples = get_scene_samples(nusc, scene_index=args.scene_index)
    print(f"Scene index: {args.scene_index}")
    print(f"Number of samples: {len(samples)}")

    sensors = get_scene_sensor_channels(nusc, scene_index=args.scene_index)
    print("Available sensors:")
    print(sensors)

    # --------------------------------------------------
    # 1. Timestamp pairing check
    # --------------------------------------------------
    print_section("1. Timestamp pairing check")

    timestamp_df = build_timestamp_table_from_scene_index(
        nusc,
        scene_index=args.scene_index,
    )

    sensors_to_eval = [s for s in sensors if s != args.ref_chan]

    pairing_summary, _ = evaluate_pairing(
        timestamp_df,
        ref_sensor=args.ref_chan,
        sensors=sensors_to_eval,
        threshold_ms=50.0,
    )

    print(pairing_summary.round(3).head(10))

    # --------------------------------------------------
    # 2. Baseline and temporal degradation
    # --------------------------------------------------
    print_section("2. Baseline and temporal degradation")

    baseline_frames = run_scene_fusion(
        nusc,
        scene_index=args.scene_index,
        radar_chan=args.radar_chan,
        ref_chan=args.ref_chan,
        min_hits=args.min_hits,
    )

    degraded_frames = run_scene_temporal_offset(
        nusc,
        scene_index=args.scene_index,
        frame_offset=-1,
    )

    baseline_summary = summarise_results(baseline_frames)
    degraded_summary = summarise_results(degraded_frames)

    print(pd.DataFrame([
        {"Method": "Baseline", **baseline_summary},
        {"Method": "Degraded (-1)", **degraded_summary},
    ]).round(3))

    # --------------------------------------------------
    # 3. Multiframe fusion
    # --------------------------------------------------
    print_section("3. Multiframe fusion")

    multiframe_frames = run_scene_multiframe_fusion(
        nusc,
        scene_index=args.scene_index,
        radar_chan=args.radar_chan,
        ref_chan=args.ref_chan,
        min_hits=args.min_hits,
    )

    weighted_frames = run_temporal_weighted_multiframe(
        nusc,
        scene_index=args.scene_index,
        radar_chan=args.radar_chan,
        ref_chan=args.ref_chan,
        min_hits=args.min_hits,
    )

    fusion_table = fusion_summary_table({
        "Baseline": baseline_frames,
        "Degraded (-1)": degraded_frames,
        "Multiframe": multiframe_frames,
        "Weighted Multiframe": weighted_frames,
    })

    print(fusion_table.round(3))

    # --------------------------------------------------
    # 4. Kalman tracking
    # --------------------------------------------------
    print_section("4. Kalman tracking")

    _, baseline_tracks = run_kalman_tracking_on_frames(baseline_frames)
    _, degraded_tracks = run_kalman_tracking_on_frames(degraded_frames)
    _, multiframe_tracks = run_kalman_tracking_on_frames(multiframe_frames)
    _, weighted_tracks = run_kalman_tracking_on_frames(weighted_frames)

    tracking_table = track_summary_table({
        "Baseline": summarise_tracks(baseline_tracks),
        "Degraded (-1)": summarise_tracks(degraded_tracks),
        "Multiframe": summarise_tracks(multiframe_tracks),
        "Weighted Multiframe": summarise_tracks(weighted_tracks),
    })

    print(tracking_table.round(3))

    # --------------------------------------------------
    # 5. Constant velocity vs constant acceleration
    # --------------------------------------------------
    print_section("5. Constant velocity vs constant acceleration")

    model_table = compare_cv_and_ca_models(
        degraded_frames,
        gate_dist=3.0,
        max_missed=2,
    )

    print(model_table.round(3))

    # --------------------------------------------------
    # 6. Spatial correction quick check
    # --------------------------------------------------
    print_section("6. Spatial correction quick check")

    # Small grid so the smoke test does not take too long.
    dx_candidates = np.linspace(-1.5, 0.5, 5)

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

    print("Best translation correction:")
    print(best_dx)

    dx_table = pd.DataFrame(all_dx)
    print(dx_table.round(3))

    # --------------------------------------------------
    # Done
    # --------------------------------------------------
    print_section("Smoke test complete")
    print("All core modules ran without crashing.")


if __name__ == "__main__":
    main()