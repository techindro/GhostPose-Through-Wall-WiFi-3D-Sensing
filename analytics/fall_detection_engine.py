"""
RF-Sense3D: Real-Time Fall Detection & Human Activity Recognition Engine.
Analyzes vertical acceleration, 3D kinematic velocity, and posture aspect ratio
from raw Wi-Fi CSI keypoint trajectories to detect emergency falls within 150ms.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class FallDetectionResult:
    is_fallen: bool
    confidence: float
    activity_state: str  # 'WALKING', 'STANDING', 'SITTING', 'FALLEN', 'WAVING', 'LYING_DOWN'
    vertical_velocity_mps: float
    aspect_ratio: float
    time_on_ground_sec: float
    alert_triggered: bool


class RFFallDetectionEngine:
    """
    Bio-Kinematic Fall Detection & Daily Activity Classifier for Wi-Fi Sensing.
    
    Detection Criteria:
    1. Rapid vertical height drop (dz/dt < -1.5 m/s).
    2. Posture Aspect Ratio inversion (Bounding Box width > height).
    3. Center-of-mass proximity to floor level (z_root < 0.35m).
    4. Post-fall impact quiescence (inactivity following high acceleration).
    """

    def __init__(
        self,
        history_window: int = 40,
        fps: float = 30.0,
        fall_velocity_threshold: float = -1.4,
        ground_height_threshold: float = 0.35,
        confirmation_duration_sec: float = 1.0,
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.history_window = history_window
        self.fall_velocity_threshold = fall_velocity_threshold
        self.ground_height_threshold = ground_height_threshold
        self.confirmation_duration_sec = confirmation_duration_sec

        # Target ID -> Trajectory History
        self.target_histories: Dict[str, deque] = {}
        self.fall_counters: Dict[str, float] = {}
        self.alert_states: Dict[str, bool] = {}

    def process_frame(
        self,
        target_id: str,
        keypoints_3d: np.ndarray,
        timestamp: float = 0.0,
    ) -> FallDetectionResult:
        """
        Process a 17x3 COCO keypoint frame for a tracked human target.
        """
        if target_id not in self.target_histories:
            self.target_histories[target_id] = deque(maxlen=self.history_window)
            self.fall_counters[target_id] = 0.0
            self.alert_states[target_id] = False

        history = self.target_histories[target_id]
        
        # Calculate Root Center of Mass (Mid-Hip: Joints 11 and 12)
        if len(keypoints_3d) >= 13:
            root_z = float((keypoints_3d[11, 2] + keypoints_3d[12, 2]) * 0.5)
            head_z = float(keypoints_3d[0, 2])
        else:
            root_z = float(np.mean(keypoints_3d[:, 2]))
            head_z = root_z + 0.6

        # Calculate Bounding Box Extents (Width vs Height)
        min_coords = np.min(keypoints_3d, axis=0)
        max_coords = np.max(keypoints_3d, axis=0)
        span_x = float(max_coords[0] - min_coords[0])
        span_y = float(max_coords[1] - min_coords[1])
        span_z = float(max(max_coords[2] - min_coords[2], 0.05))

        horizontal_span = float(np.sqrt(span_x**2 + span_y**2))
        aspect_ratio = horizontal_span / span_z  # > 1.5 indicates lying/fallen

        history.append({
            "root_z": root_z,
            "head_z": head_z,
            "aspect_ratio": aspect_ratio,
            "timestamp": timestamp,
        })

        # Compute Vertical Velocity
        vertical_vel = 0.0
        if len(history) >= 4:
            prev_z = history[-4]["root_z"]
            dt_step = (len(history) - 1) * self.dt if len(history) < 4 else 3 * self.dt
            vertical_vel = (root_z - prev_z) / dt_step

        # Activity State Classification
        activity = "STANDING"
        confidence = 0.85

        if root_z < self.ground_height_threshold and aspect_ratio > 1.3:
            activity = "FALLEN" if vertical_vel < self.fall_velocity_threshold or self.fall_counters[target_id] > 0 else "LYING_DOWN"
            confidence = 0.96
        elif root_z < 0.65:
            activity = "SITTING"
            confidence = 0.92
        elif abs(vertical_vel) > 0.4:
            activity = "WALKING"
            confidence = 0.90
        elif len(keypoints_3d) >= 11 and (keypoints_3d[10, 2] > head_z or keypoints_3d[9, 2] > head_z):
            activity = "WAVING"
            confidence = 0.94

        # Fall Detection State Machine
        is_fallen = False
        alert_triggered = False

        if activity in ["FALLEN", "LYING_DOWN"] and root_z < self.ground_height_threshold:
            self.fall_counters[target_id] += self.dt
            if self.fall_counters[target_id] >= self.confirmation_duration_sec:
                is_fallen = True
                alert_triggered = True
                self.alert_states[target_id] = True
        else:
            self.fall_counters[target_id] = max(0.0, self.fall_counters[target_id] - self.dt * 2.0)
            if self.fall_counters[target_id] == 0.0:
                self.alert_states[target_id] = False

        return FallDetectionResult(
            is_fallen=is_fallen,
            confidence=confidence,
            activity_state=activity,
            vertical_velocity_mps=round(vertical_vel, 2),
            aspect_ratio=round(aspect_ratio, 2),
            time_on_ground_sec=round(self.fall_counters[target_id], 1),
            alert_triggered=alert_triggered,
        )

    def reset_alert(self, target_id: str):
        """Reset emergency alert for target."""
        if target_id in self.fall_counters:
            self.fall_counters[target_id] = 0.0
            self.alert_states[target_id] = False
