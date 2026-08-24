"""
multi_target_tracker.py
=============================================================================
Multi-Target 3D Kalman Filter & Re-ID Associator (MIT RF-Pose Specification):
- Simultaneously tracks up to 5 distinct human targets in 3D metric space
- Mitigates identity swaps during trajectory intersections using RF Re-ID embeddings
- Implements 3D Constant Velocity Kalman Filter per target
- Associates detections via Hungarian Algorithm (Munkres / SciPy Linear Sum Assignment)
=============================================================================
"""

import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Dict, Optional, Tuple


class KalmanTrack3D:
    """
    State vector: [x, y, z, vx, vy, vz]^T
    Measurement vector: [x, y, z]^T
    """
    def __init__(self, track_id: int, initial_pos_3d: np.ndarray, reid_feature: np.ndarray):
        self.track_id = track_id
        self.reid_feature = reid_feature / (np.linalg.norm(reid_feature) + 1e-6)
        
        # State: [x, y, z, vx, vy, vz]
        self.x = np.zeros(6, dtype=np.float32)
        self.x[:3] = initial_pos_3d

        # State Transition Matrix F
        self.dt = 0.033  # ~30 FPS default
        self.F = np.eye(6, dtype=np.float32)
        for i in range(3):
            self.F[i, i + 3] = self.dt

        # Measurement Matrix H (observes [x, y, z])
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Covariance Matrices
        self.P = np.eye(6, dtype=np.float32) * 1.0
        self.Q = np.eye(6, dtype=np.float32) * 0.05  # Process noise
        self.R = np.eye(3, dtype=np.float32) * 0.1   # Measurement noise

        self.hits = 1
        self.time_since_update = 0
        self.last_skeleton_3d = None

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self.x[:3]

    def update(self, pos_3d: np.ndarray, skeleton_3d: np.ndarray, reid_feature: np.ndarray):
        y = pos_3d - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

        # Exponential Moving Average update for RF Re-ID signature (momentum=0.9)
        norm_feat = reid_feature / (np.linalg.norm(reid_feature) + 1e-6)
        self.reid_feature = 0.9 * self.reid_feature + 0.1 * norm_feat
        self.reid_feature /= (np.linalg.norm(self.reid_feature) + 1e-6)

        self.last_skeleton_3d = skeleton_3d
        self.hits += 1
        self.time_since_update = 0

    def get_position(self) -> np.ndarray:
        return self.x[:3]


class MultiTargetTracker3D:
    """
    Manages active 3D Kalman tracks, cost matrix formulation (Spatial Distance + Re-ID Cosine Distance),
    and prevents identity swaps for up to max_targets simultaneous persons.
    """
    def __init__(
        self,
        max_targets: int = 5,
        max_disappeared_frames: int = 15,
        dist_threshold_meters: float = 1.5,
        reid_weight: float = 0.6
    ):
        self.max_targets = max_targets
        self.max_disappeared = max_disappeared_frames
        self.dist_threshold = dist_threshold_meters
        self.reid_weight = reid_weight
        
        self.tracks: List[KalmanTrack3D] = []
        self.next_track_id = 1

    def update(
        self,
        detections_skeleton_3d: List[np.ndarray],  # List of [17, 3] arrays
        reid_features: List[np.ndarray]            # List of [128] arrays
    ) -> List[Dict]:
        """
        Ingests multi-person 3D skeleton detections and updates active tracks.
        """
        # 1. Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()

        num_tracks = len(self.tracks)
        num_dets = len(detections_skeleton_3d)

        # Calculate centroids (torso center: mid-hip / mid-shoulder)
        det_centroids = []
        for skel in detections_skeleton_3d:
            # Midpoint of hips (Joints 11 and 12) or global centroid
            centroid = (skel[11] + skel[12]) * 0.5
            det_centroids.append(centroid)

        if num_tracks == 0:
            # Initialize new tracks for all detections up to max_targets
            for i in range(min(num_dets, self.max_targets)):
                new_track = KalmanTrack3D(self.next_track_id, det_centroids[i], reid_features[i])
                new_track.last_skeleton_3d = detections_skeleton_3d[i]
                self.tracks.append(new_track)
                self.next_track_id += 1
        elif num_dets > 0:
            # 2. Build Hybrid Cost Matrix: Spatial Euclidean Distance + Re-ID Cosine Distance
            cost_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
            
            for t_idx, track in enumerate(self.tracks):
                track_pos = track.get_position()
                track_reid = track.reid_feature
                
                for d_idx in range(num_dets):
                    spatial_dist = np.linalg.norm(track_pos - det_centroids[d_idx])
                    # Cosine distance: 1.0 - dot_product
                    det_reid = reid_features[d_idx] / (np.linalg.norm(reid_features[d_idx]) + 1e-6)
                    cosine_dist = 1.0 - float(np.dot(track_reid, det_reid))
                    
                    # Combined Weighted Cost
                    cost_matrix[t_idx, d_idx] = (
                        (1.0 - self.reid_weight) * spatial_dist +
                        self.reid_weight * (cosine_dist * 2.0)
                    )

            # 3. Solve Assignment using Hungarian Algorithm
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_tracks = set()
            assigned_dets = set()

            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < self.dist_threshold:
                    self.tracks[r].update(det_centroids[c], detections_skeleton_3d[c], reid_features[c])
                    assigned_tracks.add(r)
                    assigned_dets.add(c)

            # 4. Spawn new tracks for unassigned detections if capacity allows
            unassigned_dets = set(range(num_dets)) - assigned_dets
            for d in unassigned_dets:
                if len(self.tracks) < self.max_targets:
                    new_track = KalmanTrack3D(self.next_track_id, det_centroids[d], reid_features[d])
                    new_track.last_skeleton_3d = detections_skeleton_3d[d]
                    self.tracks.append(new_track)
                    self.next_track_id += 1

        # 5. Prune stale / disappeared tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_disappeared]

        # Format output payload
        active_tracked_objects = []
        for t in self.tracks:
            active_tracked_objects.append({
                "track_id": f"human_{t.track_id:02d}",
                "position_3d": t.get_position().tolist(),
                "skeleton_3d": t.last_skeleton_3d.tolist() if t.last_skeleton_3d is not None else None,
                "hits": t.hits,
                "time_since_update": t.time_since_update
            })

        return active_tracked_objects
