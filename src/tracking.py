"""
tracking.py

Kalman tracking utilities for fused LiDAR-radar measurements.

This file works on frame outputs created by fusion.py / multiframe.py.
Each frame should look like:

    {
        "frame_index": int,
        "timestamp": int,   # nuScenes timestamp in microseconds
        "objects": [
            {
                "position_xy": np.array([x, y]),
                "velocity_xy": np.array([vx, vy]),
                "radar_hits": int,
                "box_name": str,
            },
            ...
        ]
    }

The main tracker is a simple constant-velocity Kalman filter:
    state = [x, y, vx, vy]

A constant-acceleration version is also included because it was tested as an
alternative model, but the project results showed that it was less stable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------
# Measurement helpers
# --------------------------------------------------

def measurement_from_object(obj: dict) -> np.ndarray:
    """
    Convert a fused object into a Kalman measurement.

    z = [x, y, vx, vy]
    """
    px, py = obj["position_xy"]
    vx, vy = obj["velocity_xy"]

    return np.array([px, py, vx, vy], dtype=float)


def measurement_covariance(
    obj: dict,
    pos_var: float = 1.0,
    vel_var_base: float = 2.0,
) -> np.ndarray:
    """
    Build a simple measurement covariance matrix.

    Position uncertainty is fixed.
    Velocity uncertainty is reduced slightly when there are more radar hits.
    This keeps the logic simple while still using radar hit count.
    """
    hits = max(int(obj.get("radar_hits", 1)), 1)
    vel_var = vel_var_base / hits

    return np.diag([pos_var, pos_var, vel_var, vel_var])


def object_speed(obj: dict) -> float:
    """
    Return speed magnitude from an object dictionary.
    """
    vx, vy = obj["velocity_xy"]
    return float(np.sqrt(vx**2 + vy**2))


# --------------------------------------------------
# Constant-velocity Kalman track
# --------------------------------------------------

class KalmanTrack:
    """
    Constant-velocity Kalman track.

    State:
        x = [px, py, vx, vy]
    """

    def __init__(
        self,
        track_id: int,
        initial_state: np.ndarray,
        timestamp: int,
        class_name: Optional[str] = None,
        initial_covariance: Optional[np.ndarray] = None,
    ):
        self.track_id = track_id
        self.x = np.asarray(initial_state, dtype=float).copy()

        if initial_covariance is None:
            self.P = np.diag([1.0, 1.0, 2.0, 2.0])
        else:
            self.P = initial_covariance.copy()

        self.last_timestamp = timestamp
        self.class_name = class_name

        self.age = 1
        self.missed = 0
        self.history = [self.x.copy()]
        self.hit_history = []

    def predict(
        self,
        dt: float,
        q_pos: float = 1.0,
        q_vel: float = 1.0,
    ) -> None:
        """
        Predict state forward using constant velocity model.
        """
        dt = max(float(dt), 1e-6)

        F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        Q = np.diag([
            q_pos * dt**2,
            q_pos * dt**2,
            q_vel * dt,
            q_vel * dt,
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement: np.ndarray, R: np.ndarray, radar_hits: int = 0) -> None:
        """
        Standard Kalman update.
        """
        z = np.asarray(measurement, dtype=float)

        H = np.eye(4)

        residual = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ residual
        self.P = (np.eye(4) - K @ H) @ self.P

        self.age += 1
        self.missed = 0
        self.history.append(self.x.copy())
        self.hit_history.append(int(radar_hits))

    def mark_missed(self) -> None:
        """
        Keep predicted state when no measurement is associated.
        """
        self.age += 1
        self.missed += 1
        self.history.append(self.x.copy())
        self.hit_history.append(0)

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[2:4]))


# --------------------------------------------------
# Data association
# --------------------------------------------------

def greedy_associate_tracks_to_measurements(
    tracks: Sequence[KalmanTrack],
    measurements: Sequence[np.ndarray],
    gate_dist: float = 3.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Greedy nearest-neighbour association in position space.

    Returns:
        matches: list of (track_index, measurement_index)
        unmatched_tracks
        unmatched_measurements

    This is simple but useful for showing how degraded measurements affect
    track formation.
    """
    if len(tracks) == 0 or len(measurements) == 0:
        return [], list(range(len(tracks))), list(range(len(measurements)))

    track_pos = np.array([track.position for track in tracks], dtype=float)
    meas_pos = np.array([meas[:2] for meas in measurements], dtype=float)

    dist_matrix = np.linalg.norm(
        track_pos[:, None, :] - meas_pos[None, :, :],
        axis=2,
    )

    matches = []
    used_tracks = set()
    used_measurements = set()

    while True:
        best_pair = None
        best_dist = np.inf

        for track_idx in range(len(tracks)):
            if track_idx in used_tracks:
                continue

            for meas_idx in range(len(measurements)):
                if meas_idx in used_measurements:
                    continue

                dist = dist_matrix[track_idx, meas_idx]

                if dist < best_dist:
                    best_dist = dist
                    best_pair = (track_idx, meas_idx)

        if best_pair is None or best_dist > gate_dist:
            break

        track_idx, meas_idx = best_pair

        matches.append((track_idx, meas_idx))
        used_tracks.add(track_idx)
        used_measurements.add(meas_idx)

    unmatched_tracks = [
        i for i in range(len(tracks))
        if i not in used_tracks
    ]

    unmatched_measurements = [
        i for i in range(len(measurements))
        if i not in used_measurements
    ]

    return matches, unmatched_tracks, unmatched_measurements


# --------------------------------------------------
# Constant-velocity tracking pipeline
# --------------------------------------------------

def run_kalman_tracking_on_frames(
    frames: Sequence[dict],
    gate_dist: float = 3.0,
    max_missed: int = 2,
    pos_var: float = 1.0,
    vel_var_base: float = 2.0,
    q_pos: float = 1.0,
    q_vel: float = 1.0,
    default_dt: float = 0.1,
) -> Tuple[List[dict], List[KalmanTrack]]:
    """
    Run constant-velocity Kalman tracking on precomputed fused frames.

    Returns:
        tracked_frames
        finished_tracks
    """
    active_tracks: List[KalmanTrack] = []
    finished_tracks: List[KalmanTrack] = []

    tracked_frames = []

    next_track_id = 0
    previous_timestamp = None

    for frame in frames:
        timestamp = int(frame.get("timestamp", 0))
        objects = frame.get("objects", [])

        if previous_timestamp is None or timestamp == 0:
            dt = default_dt
        else:
            dt = (timestamp - previous_timestamp) / 1e6

        # Predict all current tracks.
        for track in active_tracks:
            track.predict(dt, q_pos=q_pos, q_vel=q_vel)

        measurements = [measurement_from_object(obj) for obj in objects]
        covariances = [
            measurement_covariance(
                obj,
                pos_var=pos_var,
                vel_var_base=vel_var_base,
            )
            for obj in objects
        ]

        matches, unmatched_tracks, unmatched_measurements = greedy_associate_tracks_to_measurements(
            active_tracks,
            measurements,
            gate_dist=gate_dist,
        )

        # Update matched tracks.
        for track_idx, meas_idx in matches:
            active_tracks[track_idx].update(
                measurements[meas_idx],
                covariances[meas_idx],
                radar_hits=objects[meas_idx].get("radar_hits", 0),
            )
            active_tracks[track_idx].last_timestamp = timestamp

        # Keep unmatched tracks alive for a few frames.
        for track_idx in unmatched_tracks:
            active_tracks[track_idx].mark_missed()
            active_tracks[track_idx].last_timestamp = timestamp

        # Start new tracks for unmatched measurements.
        for meas_idx in unmatched_measurements:
            obj = objects[meas_idx]
            new_track = KalmanTrack(
                track_id=next_track_id,
                initial_state=measurements[meas_idx],
                timestamp=timestamp,
                class_name=obj.get("box_name"),
            )
            new_track.hit_history.append(obj.get("radar_hits", 0))

            active_tracks.append(new_track)
            next_track_id += 1

        # Move dead tracks to finished list.
        still_active = []

        for track in active_tracks:
            if track.missed > max_missed:
                finished_tracks.append(track)
            else:
                still_active.append(track)

        active_tracks = still_active

        tracked_frames.append({
            "frame_index": frame.get("frame_index"),
            "timestamp": timestamp,
            "tracks": [
                {
                    "track_id": track.track_id,
                    "state": track.x.copy(),
                    "class_name": track.class_name,
                    "missed": track.missed,
                    "age": track.age,
                }
                for track in active_tracks
            ],
            "objects": objects,
        })

        previous_timestamp = timestamp

    finished_tracks.extend(active_tracks)

    return tracked_frames, finished_tracks


def run_kalman_predicted_state_recovery_on_frames(
    frames: Sequence[dict],
    gate_dist: float = 3.0,
    max_missed: int = 2,
    pos_var: float = 1.0,
    vel_var_base: float = 2.0,
    q_pos: float = 1.0,
    q_vel: float = 1.0,
    default_dt: float = 0.1,
) -> Tuple[List[dict], List[KalmanTrack]]:
    """
    Run Kalman tracking but save predicted states BEFORE measurement update.

    This was used to test whether Kalman prediction could act as a temporal
    compensation method.
    """
    active_tracks: List[KalmanTrack] = []
    finished_tracks: List[KalmanTrack] = []

    predicted_frames = []
    next_track_id = 0
    previous_timestamp = None

    for frame in frames:
        timestamp = int(frame.get("timestamp", 0))
        objects = frame.get("objects", [])

        if previous_timestamp is None or timestamp == 0:
            dt = default_dt
        else:
            dt = (timestamp - previous_timestamp) / 1e6

        for track in active_tracks:
            track.predict(dt, q_pos=q_pos, q_vel=q_vel)

        # Save prediction snapshot before updates.
        predicted_frames.append({
            "frame_index": frame.get("frame_index"),
            "timestamp": timestamp,
            "predicted_tracks": [
                {
                    "track_id": track.track_id,
                    "state": track.x.copy(),
                    "class_name": track.class_name,
                    "missed": track.missed,
                    "age": track.age,
                }
                for track in active_tracks
            ],
        })

        measurements = [measurement_from_object(obj) for obj in objects]
        covariances = [
            measurement_covariance(
                obj,
                pos_var=pos_var,
                vel_var_base=vel_var_base,
            )
            for obj in objects
        ]

        matches, unmatched_tracks, unmatched_measurements = greedy_associate_tracks_to_measurements(
            active_tracks,
            measurements,
            gate_dist=gate_dist,
        )

        for track_idx, meas_idx in matches:
            active_tracks[track_idx].update(
                measurements[meas_idx],
                covariances[meas_idx],
                radar_hits=objects[meas_idx].get("radar_hits", 0),
            )
            active_tracks[track_idx].last_timestamp = timestamp

        for track_idx in unmatched_tracks:
            active_tracks[track_idx].mark_missed()
            active_tracks[track_idx].last_timestamp = timestamp

        for meas_idx in unmatched_measurements:
            obj = objects[meas_idx]
            new_track = KalmanTrack(
                track_id=next_track_id,
                initial_state=measurements[meas_idx],
                timestamp=timestamp,
                class_name=obj.get("box_name"),
            )
            new_track.hit_history.append(obj.get("radar_hits", 0))

            active_tracks.append(new_track)
            next_track_id += 1

        still_active = []

        for track in active_tracks:
            if track.missed > max_missed:
                finished_tracks.append(track)
            else:
                still_active.append(track)

        active_tracks = still_active
        previous_timestamp = timestamp

    finished_tracks.extend(active_tracks)

    return predicted_frames, finished_tracks


# --------------------------------------------------
# Track summaries and conversion helpers
# --------------------------------------------------

def summarise_tracks(tracks: Sequence[KalmanTrack]) -> Dict[str, float]:
    """
    Compute basic track-level summary statistics.
    """
    lengths = [len(track.history) for track in tracks]
    speeds = [track.speed for track in tracks]

    return {
        "num_tracks": len(tracks),
        "avg_track_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_track_length": float(np.median(lengths)) if lengths else 0.0,
        "avg_final_speed": float(np.mean(speeds)) if speeds else 0.0,
        "median_final_speed": float(np.median(speeds)) if speeds else 0.0,
    }


def summarise_tracked_frames(tracked_frames: Sequence[dict]) -> Dict[str, float]:
    """
    Summarise number of active track states per frame.
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


def tracked_frames_to_object_like(tracked_frames: Sequence[dict]) -> List[dict]:
    """
    Convert tracked frames into object-like frame dictionaries.

    This makes it possible to reuse measurement summary functions on track
    states if needed.
    """
    object_frames = []

    for frame in tracked_frames:
        objects = []

        for track in frame.get("tracks", []):
            x, y, vx, vy = track["state"]

            objects.append({
                "box_index": track["track_id"],
                "box_name": track.get("class_name", "track"),
                "position_xy": np.array([x, y], dtype=float),
                "velocity_xy": np.array([vx, vy], dtype=float),
                "radar_hits": 0,
                "associated_radar_points_xy": [],
            })

        object_frames.append({
            "frame_index": frame.get("frame_index"),
            "timestamp": frame.get("timestamp"),
            "objects": objects,
        })

    return object_frames


def predicted_frames_to_object_like(predicted_frames: Sequence[dict]) -> List[dict]:
    """
    Convert predicted Kalman frames into object-like frames.
    """
    object_frames = []

    for frame in predicted_frames:
        objects = []

        for track in frame.get("predicted_tracks", []):
            x, y, vx, vy = track["state"]

            objects.append({
                "box_index": track["track_id"],
                "box_name": track.get("class_name", "track"),
                "position_xy": np.array([x, y], dtype=float),
                "velocity_xy": np.array([vx, vy], dtype=float),
                "radar_hits": 0,
                "associated_radar_points_xy": [],
            })

        object_frames.append({
            "frame_index": frame.get("frame_index"),
            "timestamp": frame.get("timestamp"),
            "objects": objects,
        })

    return object_frames


def summarise_object_like_frames(frames: Sequence[dict]) -> Dict[str, float]:
    """
    Summarise object-like frame outputs.

    This is useful when comparing Kalman predicted states with fused objects.
    """
    n_frames = len(frames)
    total_objects = sum(len(frame.get("objects", [])) for frame in frames)

    speeds = []

    for frame in frames:
        for obj in frame.get("objects", []):
            speeds.append(object_speed(obj))

    return {
        "total_frames": n_frames,
        "total_objects": total_objects,
        "objects_per_frame": total_objects / n_frames if n_frames else 0.0,
        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
        "median_speed": float(np.median(speeds)) if speeds else 0.0,
    }


def track_summary_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Convert several track summaries into a table.

    Example:
        track_summary_table({
            "Baseline": summarise_tracks(baseline_tracks),
            "Degraded": summarise_tracks(degraded_tracks),
        })
    """
    rows = []

    for method, summary in results.items():
        rows.append({
            "Method": method,
            **summary,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# Constant-acceleration Kalman track
# --------------------------------------------------

class ConstantAccelerationKalmanTrack:
    """
    Constant-acceleration Kalman track.

    State:
        x = [px, py, vx, vy, ax, ay]

    Measurement:
        z = [px, py, vx, vy]
    """

    def __init__(
        self,
        track_id: int,
        initial_measurement: np.ndarray,
        timestamp: int,
        class_name: Optional[str] = None,
    ):
        px, py, vx, vy = np.asarray(initial_measurement, dtype=float)

        self.track_id = track_id
        self.x = np.array([px, py, vx, vy, 0.0, 0.0], dtype=float)
        self.P = np.diag([1.0, 1.0, 2.0, 2.0, 5.0, 5.0])

        self.last_timestamp = timestamp
        self.class_name = class_name

        self.age = 1
        self.missed = 0
        self.history = [self.x.copy()]

    def predict(self, dt: float, q: float = 1.0) -> None:
        """
        Predict using constant acceleration model.
        """
        dt = max(float(dt), 1e-6)

        F = np.array([
            [1.0, 0.0, dt,  0.0, 0.5 * dt**2, 0.0],
            [0.0, 1.0, 0.0, dt,  0.0,         0.5 * dt**2],
            [0.0, 0.0, 1.0, 0.0, dt,          0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0,         dt],
            [0.0, 0.0, 0.0, 0.0, 1.0,         0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0,         1.0],
        ])

        Q = np.eye(6) * q

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement: np.ndarray, R: Optional[np.ndarray] = None) -> None:
        """
        Update using measurement z = [px, py, vx, vy].
        """
        z = np.asarray(measurement, dtype=float)

        H = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ])

        if R is None:
            R = np.diag([1.0, 1.0, 2.0, 2.0])

        residual = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ residual
        self.P = (np.eye(6) - K @ H) @ self.P

        self.age += 1
        self.missed = 0
        self.history.append(self.x.copy())

    def mark_missed(self) -> None:
        self.age += 1
        self.missed += 1
        self.history.append(self.x.copy())

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4]

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[2:4]))


def _associate_ca_tracks_to_measurements(
    tracks: Sequence[ConstantAccelerationKalmanTrack],
    measurements: Sequence[np.ndarray],
    gate_dist: float = 3.0,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Same greedy association, but for constant-acceleration tracks.
    """
    if len(tracks) == 0 or len(measurements) == 0:
        return [], list(range(len(tracks))), list(range(len(measurements)))

    track_pos = np.array([track.position for track in tracks], dtype=float)
    meas_pos = np.array([meas[:2] for meas in measurements], dtype=float)

    dist_matrix = np.linalg.norm(
        track_pos[:, None, :] - meas_pos[None, :, :],
        axis=2,
    )

    matches = []
    used_tracks = set()
    used_measurements = set()

    while True:
        best_pair = None
        best_dist = np.inf

        for track_idx in range(len(tracks)):
            if track_idx in used_tracks:
                continue

            for meas_idx in range(len(measurements)):
                if meas_idx in used_measurements:
                    continue

                dist = dist_matrix[track_idx, meas_idx]

                if dist < best_dist:
                    best_dist = dist
                    best_pair = (track_idx, meas_idx)

        if best_pair is None or best_dist > gate_dist:
            break

        track_idx, meas_idx = best_pair
        matches.append((track_idx, meas_idx))

        used_tracks.add(track_idx)
        used_measurements.add(meas_idx)

    unmatched_tracks = [
        i for i in range(len(tracks))
        if i not in used_tracks
    ]

    unmatched_measurements = [
        i for i in range(len(measurements))
        if i not in used_measurements
    ]

    return matches, unmatched_tracks, unmatched_measurements


def run_constant_acceleration_tracking_on_frames(
    frames: Sequence[dict],
    gate_dist: float = 3.0,
    max_missed: int = 2,
    q: float = 1.0,
    default_dt: float = 0.1,
) -> List[ConstantAccelerationKalmanTrack]:
    """
    Run the constant-acceleration Kalman model on fused frames.

    This is included mainly for comparison with the constant-velocity model.
    """
    active_tracks: List[ConstantAccelerationKalmanTrack] = []
    finished_tracks: List[ConstantAccelerationKalmanTrack] = []

    next_track_id = 0
    previous_timestamp = None

    for frame in frames:
        timestamp = int(frame.get("timestamp", 0))
        objects = frame.get("objects", [])

        if previous_timestamp is None or timestamp == 0:
            dt = default_dt
        else:
            dt = (timestamp - previous_timestamp) / 1e6

        for track in active_tracks:
            track.predict(dt, q=q)

        measurements = [measurement_from_object(obj) for obj in objects]

        matches, unmatched_tracks, unmatched_measurements = _associate_ca_tracks_to_measurements(
            active_tracks,
            measurements,
            gate_dist=gate_dist,
        )

        for track_idx, meas_idx in matches:
            active_tracks[track_idx].update(measurements[meas_idx])
            active_tracks[track_idx].last_timestamp = timestamp

        for track_idx in unmatched_tracks:
            active_tracks[track_idx].mark_missed()
            active_tracks[track_idx].last_timestamp = timestamp

        for meas_idx in unmatched_measurements:
            obj = objects[meas_idx]

            new_track = ConstantAccelerationKalmanTrack(
                track_id=next_track_id,
                initial_measurement=measurements[meas_idx],
                timestamp=timestamp,
                class_name=obj.get("box_name"),
            )

            active_tracks.append(new_track)
            next_track_id += 1

        still_active = []

        for track in active_tracks:
            if track.missed > max_missed:
                finished_tracks.append(track)
            else:
                still_active.append(track)

        active_tracks = still_active
        previous_timestamp = timestamp

    finished_tracks.extend(active_tracks)

    return finished_tracks


def summarise_constant_acceleration_tracks(
    tracks: Sequence[ConstantAccelerationKalmanTrack],
) -> Dict[str, float]:
    """
    Summarise constant-acceleration tracks.
    """
    lengths = [len(track.history) for track in tracks]
    speeds = [track.speed for track in tracks]

    return {
        "num_tracks": len(tracks),
        "avg_track_length": float(np.mean(lengths)) if lengths else 0.0,
        "median_track_length": float(np.median(lengths)) if lengths else 0.0,
        "avg_speed": float(np.mean(speeds)) if speeds else 0.0,
        "median_speed": float(np.median(speeds)) if speeds else 0.0,
    }


# --------------------------------------------------
# Comparison helpers
# --------------------------------------------------

def compare_cv_and_ca_models(
    frames: Sequence[dict],
    gate_dist: float = 3.0,
    max_missed: int = 2,
) -> pd.DataFrame:
    """
    Compare constant-velocity and constant-acceleration Kalman models.
    """
    _, cv_tracks = run_kalman_tracking_on_frames(
        frames,
        gate_dist=gate_dist,
        max_missed=max_missed,
    )

    ca_tracks = run_constant_acceleration_tracking_on_frames(
        frames,
        gate_dist=gate_dist,
        max_missed=max_missed,
    )

    cv_summary = summarise_tracks(cv_tracks)
    ca_summary = summarise_constant_acceleration_tracks(ca_tracks)

    return pd.DataFrame([
        {
            "Model": "Constant-velocity Kalman",
            "Number of tracks": cv_summary["num_tracks"],
            "Average track length": cv_summary["avg_track_length"],
            "Average speed": cv_summary["avg_final_speed"],
        },
        {
            "Model": "Constant-acceleration Kalman",
            "Number of tracks": ca_summary["num_tracks"],
            "Average track length": ca_summary["avg_track_length"],
            "Average speed": ca_summary["avg_speed"],
        },
    ])


if __name__ == "__main__":
    print("tracking.py contains Kalman tracking and track-level evaluation methods.")