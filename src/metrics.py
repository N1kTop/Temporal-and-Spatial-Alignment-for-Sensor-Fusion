"""
metrics.py

Summary metrics and table helpers for the sensor fusion project.

This file keeps measurement-level, physical-alignment, and track-level
summary functions in one place.

The functions are intentionally simple because the project focuses on
interpretable evaluation rather than complex benchmarking.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------
# Basic object / frame metrics
# --------------------------------------------------

def object_speed(obj: dict) -> float:
    """
    Compute speed magnitude from an object dictionary.

    Expected object format:
        obj["velocity_xy"] = [vx, vy]
    """
    vx, vy = obj["velocity_xy"]
    return float(np.sqrt(vx**2 + vy**2))


def count_objects(frames: Sequence[dict]) -> int:
    """
    Count total objects across a list of frame dictionaries.
    """
    return int(sum(len(frame.get("objects", [])) for frame in frames))


def speeds_from_frames(frames: Sequence[dict]) -> List[float]:
    """
    Extract speed magnitudes from all objects in all frames.
    """
    speeds = []

    for frame in frames:
        for obj in frame.get("objects", []):
            speeds.append(object_speed(obj))

    return speeds


def hits_from_frames(frames: Sequence[dict]) -> List[int]:
    """
    Extract radar hit counts from all objects in all frames.
    """
    hits = []

    for frame in frames:
        for obj in frame.get("objects", []):
            hits.append(int(obj.get("radar_hits", 0)))

    return hits


def summarise_results(frames: Sequence[dict]) -> Dict[str, float]:
    """
    Compute object-level fusion summary metrics.

    This is used throughout the project for baseline, degraded,
    interpolation, and multiframe results.

    Returns both 'total_fused_objects' and 'total_objects' because some
    earlier notebook sections used different names.
    """
    total_frames = len(frames)
    total_objects = count_objects(frames)

    hits = hits_from_frames(frames)
    speeds = speeds_from_frames(frames)

    return {
        "total_frames": total_frames,
        "total_fused_objects": total_objects,
        "total_objects": total_objects,
        "objects_per_frame": total_objects / total_frames if total_frames else 0.0,

        "avg_hits": float(np.mean(hits)) if hits else 0.0,
        "median_hits": float(np.median(hits)) if hits else 0.0,

        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
        "median_speed": float(np.median(speeds)) if speeds else 0.0,
    }


def summarise_multiple_methods(method_frames: Dict[str, Sequence[dict]]) -> pd.DataFrame:
    """
    Build a summary table for multiple fusion methods.

    Example:
        summarise_multiple_methods({
            "Baseline": baseline_frames,
            "Degraded": degraded_frames,
            "Multiframe": multiframe_frames,
        })
    """
    rows = []

    for method, frames in method_frames.items():
        rows.append({
            "Method": method,
            **summarise_results(frames),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Physical alignment metrics
# --------------------------------------------------

def compute_physical_alignment_metrics(frames: Sequence[dict]) -> Dict[str, float]:
    """
    Compute physical alignment metrics from associated radar points.

    Each object should contain:
        position_xy
        associated_radar_points_xy

    Metrics:
        - distance between associated radar points and LiDAR box centre
        - number of radar points per fused object
    """
    centre_distances = []
    points_per_object = []

    for frame in frames:
        for obj in frame.get("objects", []):
            if "position_xy" not in obj:
                continue

            box_center = np.asarray(obj["position_xy"], dtype=float)

            radar_points_xy = np.asarray(
                obj.get("associated_radar_points_xy", []),
                dtype=float,
            )

            if radar_points_xy.size == 0:
                continue

            radar_points_xy = radar_points_xy.reshape(-1, 2)

            dists = np.linalg.norm(
                radar_points_xy - box_center[None, :],
                axis=1,
            )

            centre_distances.extend(dists.tolist())
            points_per_object.append(len(radar_points_xy))

    return {
        "mean_centre_distance": float(np.mean(centre_distances)) if centre_distances else np.nan,
        "median_centre_distance": float(np.median(centre_distances)) if centre_distances else np.nan,
        "max_centre_distance": float(np.max(centre_distances)) if centre_distances else np.nan,

        "mean_points_per_object": float(np.mean(points_per_object)) if points_per_object else np.nan,
        "median_points_per_object": float(np.median(points_per_object)) if points_per_object else np.nan,
        "total_associated_points": int(np.sum(points_per_object)) if points_per_object else 0,
    }


def summarise_physical_multiple(method_frames: Dict[str, Sequence[dict]]) -> pd.DataFrame:
    """
    Build a physical-alignment summary table for several methods.
    """
    rows = []

    for method, frames in method_frames.items():
        rows.append({
            "Method": method,
            **compute_physical_alignment_metrics(frames),
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Alignment error metrics
# --------------------------------------------------

def summarise_alignment_errors(values: Sequence[float]) -> Dict[str, float]:
    """
    Summarise an array-like alignment error sequence.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "n": 0,
        }

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "n": int(len(values)),
    }


def summarise_alignment_dataframe(df: pd.DataFrame,
                                  col: str = "alignment_error_xy") -> Dict[str, float]:
    """
    Summarise an alignment-error column from a dataframe.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataframe.")

    return summarise_alignment_errors(df[col].to_numpy())


def recovery_pct(clean: float,
                 degraded: float,
                 corrected: float,
                 eps: float = 1e-6,
                 clip: bool = True) -> float:
    """
    Percentage recovery from degraded to corrected.

    Formula:
        100 * (degraded - corrected) / (degraded - clean)

    If degraded and clean are almost equal, the denominator is protected.
    """
    denom = max(abs(degraded - clean), eps)
    value = 100.0 * (degraded - corrected) / denom

    if clip:
        value = np.clip(value, -100.0, 200.0)

    return float(value)


# --------------------------------------------------
# Timestamp pairing metrics
# --------------------------------------------------

def summarise_pairing_matches(match_df: pd.DataFrame) -> Dict[str, float]:
    """
    Summarise timestamp pairing results.

    Expected columns:
        matched
        dt_ms_abs
    """
    if len(match_df) == 0:
        return {
            "match_rate_pct": 0.0,
            "mean_abs_dt_ms": np.nan,
            "median_abs_dt_ms": np.nan,
            "p95_abs_dt_ms": np.nan,
            "max_abs_dt_ms": np.nan,
        }

    matched = match_df["matched"].to_numpy(dtype=bool)
    abs_dt = match_df["dt_ms_abs"].to_numpy(dtype=float)

    if not np.any(matched):
        return {
            "match_rate_pct": 0.0,
            "mean_abs_dt_ms": np.nan,
            "median_abs_dt_ms": np.nan,
            "p95_abs_dt_ms": np.nan,
            "max_abs_dt_ms": np.nan,
        }

    return {
        "match_rate_pct": float(np.mean(matched) * 100.0),
        "mean_abs_dt_ms": float(np.nanmean(abs_dt)),
        "median_abs_dt_ms": float(np.nanmedian(abs_dt)),
        "p95_abs_dt_ms": float(np.nanpercentile(abs_dt, 95)),
        "max_abs_dt_ms": float(np.nanmax(abs_dt)),
    }


def compare_pairing_summaries(summary_clean: pd.DataFrame,
                              summary_corrupt: pd.DataFrame,
                              sensor_col: str = "sensor") -> pd.DataFrame:
    """
    Combine clean and corrupted timestamp pairing summaries.

    This is mainly for cleaner result tables.
    """
    clean = summary_clean.copy()
    corrupt = summary_corrupt.copy()

    clean = clean.add_prefix("clean_")
    corrupt = corrupt.add_prefix("corrupt_")

    clean = clean.rename(columns={f"clean_{sensor_col}": sensor_col})
    corrupt = corrupt.rename(columns={f"corrupt_{sensor_col}": sensor_col})

    return clean.merge(corrupt, on=sensor_col, how="outer")


# --------------------------------------------------
# Track metrics
# --------------------------------------------------

def summarise_tracks(tracks: Sequence) -> Dict[str, float]:
    """
    Summarise a list of KalmanTrack-like objects.

    Expected track attributes:
        history
        x

    This duplicates a small amount of tracking.py logic deliberately, so
    metrics.py can be used independently in notebooks.
    """
    lengths = []
    speeds = []

    for track in tracks:
        history = getattr(track, "history", [])
        x = getattr(track, "x", None)

        lengths.append(len(history))

        if x is not None and len(x) >= 4:
            speeds.append(float(np.linalg.norm(x[2:4])))

    return {
        "num_tracks": len(tracks),
        "avg_track_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_track_length": float(np.median(lengths)) if lengths else 0.0,
        "avg_final_speed": float(np.mean(speeds)) if speeds else 0.0,
        "median_final_speed": float(np.median(speeds)) if speeds else 0.0,
    }


def summarise_tracked_frames(tracked_frames: Sequence[dict]) -> Dict[str, float]:
    """
    Summarise active track states per frame.
    """
    n_frames = len(tracked_frames)

    states_per_frame = [
        len(frame.get("tracks", []))
        for frame in tracked_frames
    ]

    total_states = int(np.sum(states_per_frame)) if states_per_frame else 0

    return {
        "total_frames": n_frames,
        "total_track_states": total_states,
        "track_states_per_frame": total_states / n_frames if n_frames else 0.0,
        "avg_active_tracks": float(np.mean(states_per_frame)) if states_per_frame else 0.0,
        "max_active_tracks": int(np.max(states_per_frame)) if states_per_frame else 0,
    }


def track_summary_table(track_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Convert track summary dictionaries into a dataframe.

    Example:
        track_summary_table({
            "Baseline": summarise_tracks(baseline_tracks),
            "Degraded": summarise_tracks(degraded_tracks),
        })
    """
    rows = []

    for method, summary in track_results.items():
        rows.append({
            "Method": method,
            **summary,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Comparison table helpers
# --------------------------------------------------

def fusion_summary_table(method_frames: Dict[str, Sequence[dict]]) -> pd.DataFrame:
    """
    Create a compact fusion summary table with common report-style columns.
    """
    rows = []

    for method, frames in method_frames.items():
        s = summarise_results(frames)

        rows.append({
            "Method": method,
            "Total frames": s["total_frames"],
            "Total fused objects": s["total_fused_objects"],
            "Objects per frame": s["objects_per_frame"],
            "Average hits": s["avg_hits"],
            "Median hits": s["median_hits"],
            "Average speed (m/s)": s["avg_speed"],
            "Median speed (m/s)": s["median_speed"],
        })

    return pd.DataFrame(rows)


def multiframe_summary_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a compact table for multiframe summary dictionaries.

    Expects entries such as:
        baseline
        degraded
        multiframe
        weighted_multiframe
    """
    rows = []

    for key, label in [
        ("baseline", "Baseline"),
        ("degraded", "Degraded"),
        ("multiframe", "Multiframe"),
        ("weighted_multiframe", "Weighted Multiframe"),
    ]:
        if key not in results:
            continue

        s = results[key]

        rows.append({
            "Method": label,
            "Objects per frame": s.get("objects_per_frame", np.nan),
            "Total objects": s.get("total_fused_objects", s.get("total_objects", np.nan)),
            "Average speed (m/s)": s.get("avg_speed", np.nan),
        })

    return pd.DataFrame(rows)


def track_level_summary_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Build a table for measurement + track-level comparison.

    This expects keys in the style used by multiframe.evaluate_multiframe_track_level:
        baseline_measurements
        baseline_tracks
        baseline_track_states
        ...
    """
    rows = []

    methods = [
        ("baseline", "Baseline"),
        ("degraded", "Degraded"),
        ("multiframe", "Naive Multiframe"),
        ("weighted_multiframe", "Weighted Multiframe"),
    ]

    for key, label in methods:
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
            "Track average speed (m/s)": tracks.get("avg_final_speed", np.nan),
        })

    return pd.DataFrame(rows)


def constant_model_comparison_table(cv_summary: Dict[str, float],
                                    ca_summary: Dict[str, float]) -> pd.DataFrame:
    """
    Build comparison table for constant-velocity vs constant-acceleration models.
    """
    return pd.DataFrame([
        {
            "Model": "Constant-velocity Kalman",
            "Number of tracks": cv_summary.get("num_tracks", np.nan),
            "Average track length": cv_summary.get("avg_track_length", np.nan),
            "Average speed (m/s)": cv_summary.get("avg_final_speed", cv_summary.get("avg_speed", np.nan)),
        },
        {
            "Model": "Constant-acceleration Kalman",
            "Number of tracks": ca_summary.get("num_tracks", np.nan),
            "Average track length": ca_summary.get("avg_track_length", np.nan),
            "Average speed (m/s)": ca_summary.get("avg_speed", ca_summary.get("avg_final_speed", np.nan)),
        },
    ])


# --------------------------------------------------
# Formatting / saving helpers
# --------------------------------------------------

def round_table(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """
    Return rounded copy of a dataframe.

    Keeps non-numeric columns unchanged.
    """
    return df.round(decimals)


def save_table_csv(df: pd.DataFrame, path: str, decimals: Optional[int] = None) -> None:
    """
    Save dataframe to CSV.

    If decimals is supplied, numeric values are rounded before saving.
    """
    out = df.copy()

    if decimals is not None:
        out = out.round(decimals)

    out.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_multiple_tables_csv(tables: Dict[str, pd.DataFrame], path: str) -> None:
    """
    Save multiple small tables into one CSV file with section labels.

    This is useful for opening all result tables in Excel without creating
    many separate files.
    """
    lines = []

    for name, df in tables.items():
        lines.append([name])
        lines.append(list(df.columns))

        for _, row in df.iterrows():
            lines.append(row.tolist())

        lines.append([])

    # Writing manually keeps section separators simple and readable.
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(lines)

    print(f"Saved: {path}")


# --------------------------------------------------
# Small sanity helpers
# --------------------------------------------------

def check_frame_format(frames: Sequence[dict], require_timestamp: bool = True) -> bool:
    """
    Basic validation for frame dictionaries.

    Returns True if the frame structure looks usable.
    Raises a ValueError if a clear problem is found.
    """
    for i, frame in enumerate(frames):
        if "objects" not in frame:
            raise ValueError(f"Frame {i} has no 'objects' key.")

        if require_timestamp and "timestamp" not in frame:
            raise ValueError(f"Frame {i} has no 'timestamp' key.")

        if not isinstance(frame["objects"], list):
            raise ValueError(f"Frame {i} 'objects' field is not a list.")

    return True


if __name__ == "__main__":
    print("metrics.py contains summary and table helper functions.")