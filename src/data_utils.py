"""
data_utils.py

Small helper functions for loading and organising nuScenes scene data.

This file is intentionally kept simple. It mainly contains utilities for:
- selecting scenes
- iterating through samples
- extracting timestamps
- building timestamp tables
- checking available sensors

Most fusion / degradation / tracking logic is kept in separate files.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Default channel names used throughout the project.
DEFAULT_REF_SENSOR = "LIDAR_TOP"
DEFAULT_RADAR_SENSOR = "RADAR_FRONT"


# nuScenes normally uses these channels.
# The code also discovers sensors automatically, so this list is mostly useful
# when a fixed order is needed for plots or tables.
DEFAULT_SENSOR_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "LIDAR_TOP",
    "RADAR_FRONT",
    "RADAR_FRONT_LEFT",
    "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT",
    "RADAR_BACK_RIGHT",
]


def get_scene_token(nusc, scene_index: Optional[int] = None,
                    scene_name: Optional[str] = None,
                    scene_token: Optional[str] = None) -> str:
    """
    Return a scene token from either:
    - an existing scene_token
    - a scene index
    - a scene name

    This avoids hard-coding the scene token in every notebook/script.
    """
    if scene_token is not None:
        return scene_token

    if scene_index is not None:
        if scene_index < 0 or scene_index >= len(nusc.scene):
            raise IndexError(f"scene_index {scene_index} is outside available scenes.")
        return nusc.scene[scene_index]["token"]

    if scene_name is not None:
        for scene in nusc.scene:
            if scene["name"] == scene_name:
                return scene["token"]
        raise ValueError(f"Scene name not found: {scene_name}")

    # Default to the first scene, which is what most experiments used.
    return nusc.scene[0]["token"]


def get_scene(nusc, scene_index: Optional[int] = None,
              scene_name: Optional[str] = None,
              scene_token: Optional[str] = None) -> dict:
    """
    Return a nuScenes scene record.
    """
    token = get_scene_token(
        nusc,
        scene_index=scene_index,
        scene_name=scene_name,
        scene_token=scene_token,
    )
    return nusc.get("scene", token)


def iter_samples_in_scene(nusc, scene_token: str) -> Iterator[dict]:
    """
    Yield all samples in a scene in temporal order.
    """
    scene = nusc.get("scene", scene_token)
    sample_token = scene["first_sample_token"]

    while sample_token:
        sample = nusc.get("sample", sample_token)
        yield sample
        sample_token = sample["next"]


def get_scene_samples(nusc, scene_index: int = 0,
                      scene_token: Optional[str] = None,
                      scene_name: Optional[str] = None) -> List[dict]:
    """
    Return all samples in a scene as a list.

    This is used by most fusion and tracking functions because random access
    to previous/next frames is often needed.
    """
    token = get_scene_token(
        nusc,
        scene_index=scene_index if scene_token is None and scene_name is None else None,
        scene_name=scene_name,
        scene_token=scene_token,
    )

    return list(iter_samples_in_scene(nusc, token))


def get_first_sample(nusc, scene_index: int = 0,
                     scene_token: Optional[str] = None,
                     scene_name: Optional[str] = None) -> dict:
    """
    Return the first sample from a scene.
    """
    samples = get_scene_samples(
        nusc,
        scene_index=scene_index,
        scene_token=scene_token,
        scene_name=scene_name,
    )

    if not samples:
        raise ValueError("Scene has no samples.")

    return samples[0]


def get_adjacent_samples(nusc, sample: dict) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Return previous and next samples for a given sample.

    Returns:
        (prev_sample, next_sample)
    """
    prev_sample = nusc.get("sample", sample["prev"]) if sample.get("prev") else None
    next_sample = nusc.get("sample", sample["next"]) if sample.get("next") else None
    return prev_sample, next_sample


def get_available_sensors(sample: dict) -> List[str]:
    """
    Return all sensor channel names available in one sample.
    """
    return list(sample["data"].keys())


def get_scene_sensor_channels(nusc, scene_index: int = 0,
                              scene_token: Optional[str] = None,
                              scene_name: Optional[str] = None) -> List[str]:
    """
    Return sensor channel names from the first sample in a scene.
    """
    sample = get_first_sample(
        nusc,
        scene_index=scene_index,
        scene_token=scene_token,
        scene_name=scene_name,
    )
    return get_available_sensors(sample)


def sample_data_record(nusc, sample: dict, channel: str) -> dict:
    """
    Get the nuScenes sample_data record for a sensor channel in a sample.
    """
    if channel not in sample["data"]:
        raise KeyError(f"Channel '{channel}' not found in sample data.")

    sd_token = sample["data"][channel]
    return nusc.get("sample_data", sd_token)


def sample_data_timestamp_us(nusc, sample_or_sd_token, channel: Optional[str] = None) -> int:
    """
    Return a sample_data timestamp in microseconds.

    Usage:
        sample_data_timestamp_us(nusc, sample, "LIDAR_TOP")
        sample_data_timestamp_us(nusc, sample_data_token)
    """
    if channel is None:
        sd = nusc.get("sample_data", sample_or_sd_token)
        return int(sd["timestamp"])

    sd = sample_data_record(nusc, sample_or_sd_token, channel)
    return int(sd["timestamp"])


def sample_timestamp_us(sample: dict) -> int:
    """
    Return the sample-level timestamp in microseconds.
    """
    return int(sample["timestamp"])


def build_scene_timestamp_table(nusc, scene_token: str) -> pd.DataFrame:
    """
    Build a timestamp table for every sample in a scene.

    Each row corresponds to one nuScenes sample.
    Columns:
        sample_token
        sample_timestamp
        one column per sensor channel, containing sample_data timestamp in us

    This table is useful for timestamp pairing and temporal alignment analysis.
    """
    rows = []

    for sample in iter_samples_in_scene(nusc, scene_token):
        row = {
            "sample_token": sample["token"],
            "sample_timestamp": int(sample["timestamp"]),
        }

        for channel, sd_token in sample["data"].items():
            sd = nusc.get("sample_data", sd_token)
            row[channel] = int(sd["timestamp"])

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("sample_timestamp")
        .reset_index(drop=True)
    )


def build_timestamp_table_from_scene_index(nusc, scene_index: int = 0) -> pd.DataFrame:
    """
    Convenience wrapper around build_scene_timestamp_table.
    """
    scene_token = get_scene_token(nusc, scene_index=scene_index)
    return build_scene_timestamp_table(nusc, scene_token)


def list_scenes(nusc, max_scenes: Optional[int] = None) -> pd.DataFrame:
    """
    Return a small table describing available scenes.
    """
    scenes = nusc.scene[:max_scenes] if max_scenes is not None else nusc.scene

    rows = []
    for i, scene in enumerate(scenes):
        rows.append({
            "scene_index": i,
            "name": scene.get("name"),
            "token": scene.get("token"),
            "description": scene.get("description"),
            "nbr_samples": scene.get("nbr_samples"),
        })

    return pd.DataFrame(rows)


def print_scenes(nusc, max_scenes: Optional[int] = 10) -> None:
    """
    Print a quick scene summary.

    Kept as a simple helper because this was useful during notebook exploration.
    """
    scenes = nusc.scene[:max_scenes] if max_scenes is not None else nusc.scene

    for i, scene in enumerate(scenes):
        print(f"[{i:02d}] {scene['name']}")
        print(f"     {scene.get('description', '')}")
        print(f"     token: {scene['token']}")
        print()


def get_sensor_timestamps(df: pd.DataFrame, sensor: str,
                          dropna: bool = True) -> np.ndarray:
    """
    Extract a sensor timestamp column from a timestamp table.

    Returns timestamps as float64 so NaN values can be represented.
    """
    if sensor not in df.columns:
        raise KeyError(f"Sensor '{sensor}' not found in timestamp table.")

    values = df[sensor].to_numpy(dtype="float64")

    if dropna:
        values = values[np.isfinite(values)]

    return values


def get_reference_timestamps(df: pd.DataFrame,
                             ref_sensor: str = DEFAULT_REF_SENSOR) -> np.ndarray:
    """
    Extract reference sensor timestamps from a timestamp table.
    """
    return get_sensor_timestamps(df, ref_sensor, dropna=True)


def validate_sensor_columns(df: pd.DataFrame,
                            sensors: Sequence[str],
                            strict: bool = False) -> List[str]:
    """
    Check which requested sensor columns exist in a timestamp table.

    If strict=True, raise an error if any sensor is missing.
    Otherwise, return the list of sensors that are available.
    """
    available = []
    missing = []

    for sensor in sensors:
        if sensor in df.columns:
            available.append(sensor)
        else:
            missing.append(sensor)

    if strict and missing:
        raise KeyError(f"Missing sensor columns: {missing}")

    return available


def nearest_timestamp_index(target_ts_us: float,
                            candidate_ts_us: np.ndarray) -> Optional[int]:
    """
    Find the index of the candidate timestamp nearest to target_ts_us.

    candidate_ts_us should already be one-dimensional.
    Returns None if there are no finite candidates.
    """
    candidates = np.asarray(candidate_ts_us, dtype="float64")
    finite_mask = np.isfinite(candidates)

    if not np.any(finite_mask):
        return None

    finite_indices = np.where(finite_mask)[0]
    finite_values = candidates[finite_mask]

    best_local = int(np.argmin(np.abs(finite_values - target_ts_us)))
    return int(finite_indices[best_local])


def nearest_timestamp_delta_ms(target_ts_us: float,
                               candidate_ts_us: np.ndarray,
                               signed: bool = True) -> float:
    """
    Return nearest timestamp difference in milliseconds.

    If signed=True:
        result = nearest_candidate - target

    If signed=False:
        result = abs(nearest_candidate - target)
    """
    idx = nearest_timestamp_index(target_ts_us, candidate_ts_us)

    if idx is None:
        return np.nan

    dt_ms = (float(candidate_ts_us[idx]) - float(target_ts_us)) / 1000.0
    return dt_ms if signed else abs(dt_ms)


def same_sample_pairing_errors(nusc, scene_index: int = 0,
                               ref_sensor: str = DEFAULT_REF_SENSOR,
                               sensors: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    """
    Compute signed same-sample timestamp differences against a reference sensor.

    This does not do nearest-neighbour matching. It simply compares timestamps
    from the same nuScenes sample row:
        dt = sensor_timestamp - reference_timestamp

    Useful for checking the nominal timing structure of nuScenes.
    """
    samples = get_scene_samples(nusc, scene_index=scene_index)

    if not samples:
        return {}

    if sensors is None:
        sensors = [s for s in get_available_sensors(samples[0]) if s != ref_sensor]

    errors = {sensor: [] for sensor in sensors if sensor != ref_sensor}

    for sample in samples:
        if ref_sensor not in sample["data"]:
            continue

        t_ref = sample_data_timestamp_us(nusc, sample, ref_sensor)

        for sensor in errors:
            if sensor not in sample["data"]:
                errors[sensor].append(np.nan)
                continue

            t_sensor = sample_data_timestamp_us(nusc, sample, sensor)
            errors[sensor].append((t_sensor - t_ref) / 1000.0)

    return {sensor: np.asarray(vals, dtype=float) for sensor, vals in errors.items()}


def collect_prev_sample_data_tokens(nusc, sample_data_token: str,
                                    max_sweeps: int = 10) -> List[str]:
    """
    Collect previous sample_data tokens for a sensor stream.

    Includes the starting sample_data_token as the first item.
    """
    tokens = []
    token = sample_data_token

    for _ in range(max_sweeps):
        if not token:
            break

        tokens.append(token)

        sd = nusc.get("sample_data", token)
        token = sd.get("prev", "")

    return tokens


def collect_prev_sweep_timestamps(nusc, sample_data_token: str,
                                  max_sweeps: int = 10) -> List[int]:
    """
    Collect timestamps from the current and previous sweeps of one sample_data stream.
    """
    timestamps = []

    for token in collect_prev_sample_data_tokens(nusc, sample_data_token, max_sweeps=max_sweeps):
        sd = nusc.get("sample_data", token)
        timestamps.append(int(sd["timestamp"]))

    return timestamps


def build_radar_sweep_pool(nusc, scene_token: str,
                           radar_chan: str = DEFAULT_RADAR_SENSOR,
                           ref_chan: str = DEFAULT_REF_SENSOR,
                           max_sweeps: int = 10,
                           window_s: float = 0.5) -> pd.DataFrame:
    """
    Build a simple pool of previous radar sweep timestamps around each reference frame.

    This is mainly used for temporal alignment experiments where one LiDAR frame
    is matched against multiple nearby radar sweeps.

    Output columns:
        frame_idx
        sample_token
        t_ref_us
        radar_ts_us_list
    """
    rows = []

    for frame_idx, sample in enumerate(iter_samples_in_scene(nusc, scene_token)):
        if ref_chan not in sample["data"] or radar_chan not in sample["data"]:
            continue

        ref_sd = nusc.get("sample_data", sample["data"][ref_chan])
        radar_sd_token = sample["data"][radar_chan]

        t_ref_us = int(ref_sd["timestamp"])
        radar_ts = collect_prev_sweep_timestamps(
            nusc,
            radar_sd_token,
            max_sweeps=max_sweeps,
        )

        # Keep only radar sweeps close to the reference timestamp.
        max_dt_us = window_s * 1e6
        radar_ts = [
            int(t)
            for t in radar_ts
            if abs(int(t) - t_ref_us) <= max_dt_us
        ]

        rows.append({
            "frame_idx": frame_idx,
            "sample_token": sample["token"],
            "t_ref_us": t_ref_us,
            "radar_ts_us_list": radar_ts,
        })

    return pd.DataFrame(rows)


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """
    Small CSV save helper.

    This is not necessary, but keeps scripts a bit cleaner.
    """
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Small CSV load helper.
    """
    return pd.read_csv(path)


if __name__ == "__main__":
    # This file is normally imported by notebooks/scripts.
    # The block below is just a lightweight reminder of expected usage.
    print("data_utils.py provides helper functions for nuScenes scene and timestamp handling.")