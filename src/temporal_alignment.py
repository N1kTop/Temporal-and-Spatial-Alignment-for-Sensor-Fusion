"""
temporal_alignment.py

Temporal alignment utilities for LiDAR-radar / multi-sensor experiments.

This file contains:
- nearest timestamp matching
- pairing error summaries
- continuity-based offset smoothing
- frame-wise offset estimation
- radar sweep-pool matching helpers

Some functions expect an external alignment-error function, usually from the
notebook, because the exact radar spread / alignment metric depends on the
experiment setup.

The expected alignment function is:

    alignment_error_fn(
        nusc,
        scene_token,
        pose_time_offset_ms=...,
        window_s=...,
        max_sweeps=...
    )

and it should return a dataframe containing an "alignment_error_xy" column.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .degradation import drift_offset_fn


OffsetValue = Union[int, float, Callable[[int], float]]


# --------------------------------------------------
# Basic timestamp matching
# --------------------------------------------------

def match_nearest_timestamps(
    ref_ts_us: np.ndarray,
    tgt_ts_us: np.ndarray,
    threshold_ms: float = 50.0,
) -> pd.DataFrame:
    """
    Match each reference timestamp to the nearest target timestamp.

    Parameters
    ----------
    ref_ts_us:
        Reference timestamps in microseconds.

    tgt_ts_us:
        Target timestamps in microseconds.

    threshold_ms:
        Maximum allowed absolute time difference.

    Returns
    -------
    DataFrame with one row per reference timestamp:
        matched
        dt_ms_signed
        dt_ms_abs
        tgt_ts_us
        chosen_idx
    """
    ref = np.asarray(ref_ts_us, dtype="float64")
    tgt = np.asarray(tgt_ts_us, dtype="float64")

    out = pd.DataFrame({
        "matched": np.zeros(len(ref), dtype=bool),
        "dt_ms_signed": np.full(len(ref), np.nan),
        "dt_ms_abs": np.full(len(ref), np.nan),
        "tgt_ts_us": np.full(len(ref), np.nan),
        "chosen_idx": np.full(len(ref), -1, dtype=int),
    })

    valid_idx = np.where(np.isfinite(tgt))[0]
    if len(valid_idx) == 0:
        return out

    tgt_valid = tgt[valid_idx]

    # Sort while keeping original indices.
    order = np.argsort(tgt_valid)
    tgt_sorted = tgt_valid[order]
    idx_sorted = valid_idx[order]

    threshold_us = threshold_ms * 1000.0

    for i, t_ref in enumerate(ref):
        if not np.isfinite(t_ref):
            continue

        pos = np.searchsorted(tgt_sorted, t_ref)

        candidate_positions = []
        if pos > 0:
            candidate_positions.append(pos - 1)
        if pos < len(tgt_sorted):
            candidate_positions.append(pos)

        if not candidate_positions:
            continue

        best_pos = min(
            candidate_positions,
            key=lambda p: abs(tgt_sorted[p] - t_ref)
        )

        best_ts = tgt_sorted[best_pos]
        dt_us = best_ts - t_ref

        if abs(dt_us) <= threshold_us:
            out.loc[i, "matched"] = True
            out.loc[i, "dt_ms_signed"] = dt_us / 1000.0
            out.loc[i, "dt_ms_abs"] = abs(dt_us) / 1000.0
            out.loc[i, "tgt_ts_us"] = best_ts
            out.loc[i, "chosen_idx"] = int(idx_sorted[best_pos])

    return out


def pairing_errors(ref_ts_us: np.ndarray,
                   tgt_ts_us: np.ndarray,
                   signed: bool = True) -> np.ndarray:
    """
    Simple nearest timestamp error array.

    This is useful for quick plots and sanity checks.
    It does not apply a threshold.
    """
    ref = np.asarray(ref_ts_us, dtype="float64")
    tgt = np.asarray(tgt_ts_us, dtype="float64")

    tgt_valid = tgt[np.isfinite(tgt)]
    if len(tgt_valid) == 0:
        return np.array([])

    errors = []

    for t_ref in ref:
        if not np.isfinite(t_ref):
            continue

        idx = np.argmin(np.abs(tgt_valid - t_ref))
        dt_ms = (tgt_valid[idx] - t_ref) / 1000.0
        errors.append(dt_ms if signed else abs(dt_ms))

    return np.asarray(errors, dtype=float)


# --------------------------------------------------
# Pairing evaluation
# --------------------------------------------------

def summarise_match_df(match_df: pd.DataFrame) -> Dict[str, float]:
    """
    Summarise a timestamp matching dataframe.
    """
    matched = match_df["matched"].to_numpy(dtype=bool)
    abs_dt = match_df["dt_ms_abs"].to_numpy(dtype="float64")

    if len(match_df) == 0:
        return {
            "match_rate_pct": 0.0,
            "mean_abs_dt_ms": np.nan,
            "p95_abs_dt_ms": np.nan,
            "max_abs_dt_ms": np.nan,
        }

    if not np.any(matched):
        return {
            "match_rate_pct": 0.0,
            "mean_abs_dt_ms": np.nan,
            "p95_abs_dt_ms": np.nan,
            "max_abs_dt_ms": np.nan,
        }

    return {
        "match_rate_pct": float(np.mean(matched) * 100.0),
        "mean_abs_dt_ms": float(np.nanmean(abs_dt)),
        "p95_abs_dt_ms": float(np.nanpercentile(abs_dt, 95)),
        "max_abs_dt_ms": float(np.nanmax(abs_dt)),
    }


def evaluate_pairing(
    timestamp_df: pd.DataFrame,
    ref_sensor: str,
    sensors: Sequence[str],
    threshold_ms: float = 50.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Evaluate nearest timestamp pairing for multiple sensors against one reference.

    Returns:
        summary dataframe
        dictionary of per-sensor match dataframes
    """
    if ref_sensor not in timestamp_df.columns:
        raise KeyError(f"Reference sensor '{ref_sensor}' not found in dataframe.")

    ref_ts = timestamp_df[ref_sensor].to_numpy(dtype="float64")

    rows = []
    per_sensor = {}

    for sensor in sensors:
        if sensor == ref_sensor:
            continue

        if sensor not in timestamp_df.columns:
            rows.append({
                "sensor": sensor,
                "threshold_ms": threshold_ms,
                "match_rate_pct": 0.0,
                "mean_abs_dt_ms": np.nan,
                "p95_abs_dt_ms": np.nan,
                "max_abs_dt_ms": np.nan,
            })
            continue

        tgt_ts = timestamp_df[sensor].to_numpy(dtype="float64")

        matches = match_nearest_timestamps(
            ref_ts,
            tgt_ts,
            threshold_ms=threshold_ms,
        )

        per_sensor[sensor] = matches

        row = {
            "sensor": sensor,
            "threshold_ms": threshold_ms,
            **summarise_match_df(matches),
        }
        rows.append(row)

    return pd.DataFrame(rows), per_sensor


def compare_clean_corrupt_pairing(
    clean_df: pd.DataFrame,
    corrupt_df: pd.DataFrame,
    ref_sensor: str,
    sensors: Sequence[str],
    threshold_ms: float = 50.0,
) -> Dict[str, object]:
    """
    Convenience wrapper for the clean vs corrupted timestamp experiment.
    """
    summary_clean, matches_clean = evaluate_pairing(
        clean_df,
        ref_sensor=ref_sensor,
        sensors=sensors,
        threshold_ms=threshold_ms,
    )

    summary_corrupt, matches_corrupt = evaluate_pairing(
        corrupt_df,
        ref_sensor=ref_sensor,
        sensors=sensors,
        threshold_ms=threshold_ms,
    )

    return {
        "summary_clean": summary_clean,
        "summary_corrupt": summary_corrupt,
        "matches_clean": matches_clean,
        "matches_corrupt": matches_corrupt,
    }


# --------------------------------------------------
# Alignment error summaries
# --------------------------------------------------

def summarise_alignment(df: pd.DataFrame,
                        col: str = "alignment_error_xy") -> Dict[str, float]:
    """
    Summarise an alignment error column.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in dataframe.")

    values = df[col].to_numpy(dtype="float64")
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            "mean": np.nan,
            "p95": np.nan,
            "max": np.nan,
            "n": 0,
        }

    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "n": int(len(values)),
    }


def recovery_pct_safe(clean: float,
                      degraded: float,
                      corrected: float,
                      eps: float = 1e-6,
                      clip: bool = True) -> float:
    """
    Recovery percentage used in the results table.

    recovery = 100 * (degraded - corrected) / (degraded - clean)

    If clean and degraded are almost identical, the denominator is protected
    to avoid division by zero.
    """
    denom = max(abs(degraded - clean), eps)
    recovery = 100.0 * (degraded - corrected) / denom

    if clip:
        recovery = np.clip(recovery, -100.0, 200.0)

    return float(recovery)


# --------------------------------------------------
# Offset estimation and smoothing
# --------------------------------------------------

def smooth_offsets(offsets: Sequence[float],
                   alpha: float = 0.25) -> np.ndarray:
    """
    Exponential smoothing for frame-wise offset estimates.

    Smaller alpha means stronger smoothing.
    """
    offsets = np.asarray(offsets, dtype="float64")

    if len(offsets) == 0:
        return offsets

    smoothed = np.zeros_like(offsets)
    smoothed[0] = offsets[0]

    for i in range(1, len(offsets)):
        smoothed[i] = alpha * offsets[i] + (1.0 - alpha) * smoothed[i - 1]

    return smoothed


def offset_velocity_metric(offsets: Sequence[float]) -> Tuple[float, float]:
    """
    Measure frame-to-frame offset movement.

    Returns:
        mean absolute offset change
        max absolute offset change
    """
    offsets = np.asarray(offsets, dtype="float64")

    if len(offsets) < 2:
        return np.nan, np.nan

    diffs = np.abs(np.diff(offsets))
    return float(np.mean(diffs)), float(np.max(diffs))


def estimate_best_offset_from_raw_errors(
    raw_errors: Dict[float, np.ndarray],
    candidate_offsets: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given raw error curves for each candidate offset, choose the best offset
    independently for each frame.
    """
    candidate_offsets = np.asarray(candidate_offsets, dtype="float64")

    errors = np.stack(
        [np.asarray(raw_errors[o], dtype="float64") for o in candidate_offsets],
        axis=1,
    )

    best_idx = np.nanargmin(errors, axis=1)

    best_offsets = candidate_offsets[best_idx]
    best_errors = errors[np.arange(errors.shape[0]), best_idx]

    return best_offsets, best_errors


def estimate_best_offset_for_scene(
    nusc,
    scene_token: str,
    candidate_offsets: Sequence[float],
    alignment_error_fn: Callable,
    error_col: str = "alignment_error_xy",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, Dict[float, np.ndarray]]:
    """
    Estimate best temporal offset for each frame by grid search.

    alignment_error_fn must return a dataframe containing error_col.

    Example:
        best_offsets, best_errors, raw_errors = estimate_best_offset_for_scene(
            nusc,
            scene_token,
            candidate_offsets=[-150, -50, 0, 50, 150],
            alignment_error_fn=compute_scene_radar_spread_timeseries,
            window_s=2.0,
            max_sweeps=40
        )
    """
    raw_errors = {}

    for offset in candidate_offsets:
        df = alignment_error_fn(
            nusc,
            scene_token,
            pose_time_offset_ms=offset,
            **kwargs,
        )

        if error_col not in df.columns:
            raise KeyError(f"alignment_error_fn output must contain '{error_col}'.")

        raw_errors[float(offset)] = df[error_col].to_numpy(dtype="float64")

    best_offsets, best_errors = estimate_best_offset_from_raw_errors(
        raw_errors,
        candidate_offsets=[float(o) for o in candidate_offsets],
    )

    return best_offsets, best_errors, raw_errors


def estimate_offsets_with_continuity(
    raw_errors: Dict[float, np.ndarray],
    candidate_offsets: Sequence[float],
    lam: float = 0.02,
) -> np.ndarray:
    """
    Greedy continuity-regularised offset estimation.

    Instead of choosing the best offset independently for every frame,
    this penalises sudden jumps from the previous chosen offset.

    cost = raw_error + lam * abs(candidate_offset - previous_offset)
    """
    candidate_offsets = np.asarray(candidate_offsets, dtype="float64")

    n_frames = len(next(iter(raw_errors.values())))
    chosen = np.zeros(n_frames, dtype="float64")

    first_errors = np.array(
        [raw_errors[float(o)][0] for o in candidate_offsets],
        dtype="float64",
    )
    chosen[0] = candidate_offsets[int(np.nanargmin(first_errors))]

    for i in range(1, n_frames):
        previous = chosen[i - 1]

        costs = []
        for offset in candidate_offsets:
            raw_error = raw_errors[float(offset)][i]
            continuity_penalty = lam * abs(offset - previous)
            costs.append(raw_error + continuity_penalty)

        chosen[i] = candidate_offsets[int(np.nanargmin(costs))]

    return chosen


# --------------------------------------------------
# Drift alignment experiment wrapper
# --------------------------------------------------

def run_drift_alignment_experiment(
    nusc,
    scene_token: str,
    alignment_error_fn: Callable,
    candidate_offsets: Sequence[float] = tuple(range(-300, 301, 50)),
    a_ms_per_frame: float = 0.5,
    b_ms: float = 0.0,
    jitter_ms: float = 5.0,
    seed: int = 42,
    alpha: float = 0.25,
    continuity_lambda: float = 0.02,
    error_col: str = "alignment_error_xy",
    **kwargs,
) -> Dict[str, object]:
    """
    Run the full drift-style temporal alignment experiment.

    This mirrors the main logic used in the notebook:
        clean
        drifted
        estimated + smoothed
        estimated + continuity + smoothed

    alignment_error_fn is kept external because the exact alignment metric
    depends on the radar-spread experiment code.
    """
    df_clean = alignment_error_fn(
        nusc,
        scene_token,
        pose_time_offset_ms=0.0,
        **kwargs,
    )

    corrupt_fn = drift_offset_fn(
        a_ms_per_frame=a_ms_per_frame,
        b_ms=b_ms,
        jitter_ms=jitter_ms,
        seed=seed,
    )

    df_drifted = alignment_error_fn(
        nusc,
        scene_token,
        pose_time_offset_ms=corrupt_fn,
        **kwargs,
    )

    best_offsets, best_errors, raw_errors = estimate_best_offset_for_scene(
        nusc,
        scene_token,
        candidate_offsets=candidate_offsets,
        alignment_error_fn=alignment_error_fn,
        error_col=error_col,
        **kwargs,
    )

    smoothed_offsets = smooth_offsets(best_offsets, alpha=alpha)

    df_smoothed = alignment_error_fn(
        nusc,
        scene_token,
        pose_time_offset_ms=lambda i: smoothed_offsets[min(i, len(smoothed_offsets) - 1)],
        **kwargs,
    )

    continuity_offsets = estimate_offsets_with_continuity(
        raw_errors,
        candidate_offsets=[float(o) for o in candidate_offsets],
        lam=continuity_lambda,
    )

    continuity_offsets = smooth_offsets(continuity_offsets, alpha=alpha)

    df_continuity = alignment_error_fn(
        nusc,
        scene_token,
        pose_time_offset_ms=lambda i: continuity_offsets[min(i, len(continuity_offsets) - 1)],
        **kwargs,
    )

    summary = pd.DataFrame([
        {"condition": "clean", **summarise_alignment(df_clean, col=error_col)},
        {"condition": "drifted", **summarise_alignment(df_drifted, col=error_col)},
        {"condition": "smoothed", **summarise_alignment(df_smoothed, col=error_col)},
        {"condition": "smoothed + continuity", **summarise_alignment(df_continuity, col=error_col)},
    ])

    return {
        "clean": df_clean,
        "drifted": df_drifted,
        "smoothed": df_smoothed,
        "continuity": df_continuity,
        "best_offsets": best_offsets,
        "best_errors": best_errors,
        "smoothed_offsets": smoothed_offsets,
        "continuity_offsets": continuity_offsets,
        "raw_errors": raw_errors,
        "summary": summary,
    }


def build_recovery_table(
    results_by_scene: Dict[int, Dict[str, object]],
) -> pd.DataFrame:
    """
    Build a multi-scene recovery table from run_drift_alignment_experiment outputs.
    """
    rows = []

    for scene_id, result in results_by_scene.items():
        summary = result["summary"]

        clean = summary.loc[summary["condition"] == "clean", "mean"].iloc[0]
        drifted = summary.loc[summary["condition"] == "drifted", "mean"].iloc[0]
        corrected = summary.loc[
            summary["condition"] == "smoothed + continuity",
            "mean"
        ].iloc[0]

        recovery = recovery_pct_safe(clean, drifted, corrected)

        rows.append({
            "scene": scene_id,
            "clean_mean_error": clean,
            "drifted_mean_error": drifted,
            "corrected_mean_error": corrected,
            "recovery_pct": recovery,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Radar sweep-pool matching
# --------------------------------------------------

def _nearest_from_list(t_ref_us: float,
                       candidates_us: Sequence[float]) -> Tuple[bool, float, float, int]:
    """
    Internal helper for sweep-pool matching.

    Returns:
        matched_like
        chosen_timestamp
        abs_dt_ms
        chosen_index_in_list
    """
    if candidates_us is None or len(candidates_us) == 0:
        return False, np.nan, np.nan, -1

    arr = np.asarray(candidates_us, dtype="float64")
    finite = np.isfinite(arr)

    if not np.any(finite):
        return False, np.nan, np.nan, -1

    valid_indices = np.where(finite)[0]
    valid_values = arr[finite]

    local_idx = int(np.argmin(np.abs(valid_values - t_ref_us)))
    chosen_list_idx = int(valid_indices[local_idx])
    chosen_ts = float(valid_values[local_idx])
    abs_dt_ms = abs(chosen_ts - t_ref_us) / 1000.0

    return True, chosen_ts, abs_dt_ms, chosen_list_idx


def match_pool_nearest(
    pool_df: pd.DataFrame,
    threshold_ms: float = 80.0,
    ref_col: str = "t_ref_us",
    candidates_col: str = "radar_ts_us_list",
) -> pd.DataFrame:
    """
    Match each reference timestamp to nearest timestamp in a local sweep pool.

    pool_df is expected to contain:
        t_ref_us
        radar_ts_us_list
    """
    rows = []

    for _, row in pool_df.iterrows():
        t_ref = float(row[ref_col])
        candidates = row[candidates_col]

        has_candidate, chosen_ts, abs_dt_ms, chosen_idx = _nearest_from_list(
            t_ref,
            candidates,
        )

        matched = bool(has_candidate and abs_dt_ms <= threshold_ms)

        rows.append({
            "matched": matched,
            "dt_ms_abs": abs_dt_ms if matched else np.nan,
            "dt_ms_signed": ((chosen_ts - t_ref) / 1000.0) if matched else np.nan,
            "chosen_ts_us": chosen_ts if matched else np.nan,
            "chosen_sweep_idx": chosen_idx if matched else -1,
        })

    return pd.DataFrame(rows)


def index_thrashing(match_df: pd.DataFrame,
                    idx_col: str = "chosen_sweep_idx") -> int:
    """
    Count how often the chosen sweep index changes between consecutive frames.

    Lower value means more temporally stable sweep selection.
    """
    if idx_col not in match_df.columns:
        return 0

    idx = match_df[idx_col].to_numpy()
    idx = idx[idx >= 0]

    if len(idx) < 2:
        return 0

    return int(np.sum(idx[1:] != idx[:-1]))


def match_pool_continuity(
    pool_df: pd.DataFrame,
    threshold_ms: float = 80.0,
    max_jump: int = 0,
    ref_col: str = "t_ref_us",
    candidates_col: str = "radar_ts_us_list",
) -> pd.DataFrame:
    """
    Continuity-constrained matching for radar sweep pools.

    max_jump controls how much the chosen sweep index is allowed to change
    from the previous frame.

    max_jump = 0 is very strict.
    max_jump = 1 or more behaves closer to normal nearest matching.
    """
    rows = []
    prev_idx = None

    for _, row in pool_df.iterrows():
        t_ref = float(row[ref_col])
        candidates = np.asarray(row[candidates_col], dtype="float64")

        if len(candidates) == 0:
            rows.append({
                "matched": False,
                "dt_ms_abs": np.nan,
                "dt_ms_signed": np.nan,
                "chosen_ts_us": np.nan,
                "chosen_sweep_idx": -1,
            })
            continue

        valid_indices = np.where(np.isfinite(candidates))[0]

        if len(valid_indices) == 0:
            rows.append({
                "matched": False,
                "dt_ms_abs": np.nan,
                "dt_ms_signed": np.nan,
                "chosen_ts_us": np.nan,
                "chosen_sweep_idx": -1,
            })
            continue

        if prev_idx is None:
            allowed_indices = valid_indices
        else:
            allowed_indices = [
                i for i in valid_indices
                if abs(i - prev_idx) <= max_jump
            ]

            # If the continuity gate is too strict, fall back to valid indices.
            # This avoids completely breaking on short / uneven sweep lists.
            if len(allowed_indices) == 0:
                allowed_indices = valid_indices

        best_idx = min(
            allowed_indices,
            key=lambda i: abs(candidates[i] - t_ref)
        )

        chosen_ts = float(candidates[best_idx])
        dt_ms = (chosen_ts - t_ref) / 1000.0
        abs_dt_ms = abs(dt_ms)

        matched = abs_dt_ms <= threshold_ms

        if matched:
            prev_idx = int(best_idx)

        rows.append({
            "matched": bool(matched),
            "dt_ms_abs": abs_dt_ms if matched else np.nan,
            "dt_ms_signed": dt_ms if matched else np.nan,
            "chosen_ts_us": chosen_ts if matched else np.nan,
            "chosen_sweep_idx": int(best_idx) if matched else -1,
        })

    return pd.DataFrame(rows)


def match_pool_soft_continuity(
    pool_df: pd.DataFrame,
    threshold_ms: float = 80.0,
    lambda_jump: float = 10.0,
    ref_col: str = "t_ref_us",
    candidates_col: str = "radar_ts_us_list",
) -> pd.DataFrame:
    """
    Soft continuity version of sweep-pool matching.

    Instead of forcing the chosen index to stay close, it adds a penalty:

        cost = abs_dt_ms + lambda_jump * abs(candidate_index - previous_index)

    This is less brittle than hard max_jump.
    """
    rows = []
    prev_idx = None

    for _, row in pool_df.iterrows():
        t_ref = float(row[ref_col])
        candidates = np.asarray(row[candidates_col], dtype="float64")

        valid_indices = np.where(np.isfinite(candidates))[0]

        if len(valid_indices) == 0:
            rows.append({
                "matched": False,
                "dt_ms_abs": np.nan,
                "dt_ms_signed": np.nan,
                "chosen_ts_us": np.nan,
                "chosen_sweep_idx": -1,
            })
            continue

        costs = []

        for idx in valid_indices:
            abs_dt_ms = abs(candidates[idx] - t_ref) / 1000.0
            jump_penalty = 0.0 if prev_idx is None else lambda_jump * abs(idx - prev_idx)
            costs.append(abs_dt_ms + jump_penalty)

        best_local = int(np.argmin(costs))
        best_idx = int(valid_indices[best_local])

        chosen_ts = float(candidates[best_idx])
        dt_ms = (chosen_ts - t_ref) / 1000.0
        abs_dt_ms = abs(dt_ms)

        matched = abs_dt_ms <= threshold_ms

        if matched:
            prev_idx = best_idx

        rows.append({
            "matched": bool(matched),
            "dt_ms_abs": abs_dt_ms if matched else np.nan,
            "dt_ms_signed": dt_ms if matched else np.nan,
            "chosen_ts_us": chosen_ts if matched else np.nan,
            "chosen_sweep_idx": best_idx if matched else -1,
        })

    return pd.DataFrame(rows)


def sweep_match_metrics(match_df: pd.DataFrame) -> Dict[str, float]:
    """
    Metrics for sweep-pool matching, including index thrashing.
    """
    metrics = summarise_match_df(match_df)
    metrics["thrash"] = index_thrashing(match_df)
    return metrics


if __name__ == "__main__":
    print("temporal_alignment.py contains timestamp pairing and temporal correction helpers.")