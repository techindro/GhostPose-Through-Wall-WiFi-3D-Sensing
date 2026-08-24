"""
roi_masking.py
=============================================================================
3D Spatial Region of Interest (ROI) & Wall Reflection Masking Calculator:
- Defines physical 3D room boundaries [x_min, x_max, y_min, y_max, z_min, z_max]
- Filters out multipath ghost artifacts and exterior out-of-boundary reflections
- Calculates 3D Axis-Aligned Bounding Boxes (AABB) for detected human skeletons
- Supports custom exclusion zones (e.g., fan/rotating machinery interference)
=============================================================================
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


class SpatialROIFilter:
    """
    3D Spatial Region-of-Interest (ROI) and Multipath Ghost Reflection Filter.
    """
    def __init__(
        self,
        room_bounds: Tuple[float, float, float, float, float, float] = (-3.0, 3.0, -1.0, 5.0, 0.0, 2.5),
        min_keypoints_in_roi: int = 12,
        max_skeleton_height_m: float = 2.2,
        min_skeleton_height_m: float = 0.6
    ):
        """
        room_bounds: (x_min, x_max, y_min, y_max, z_min, z_max) in meters.
        """
        self.x_min, self.x_max = room_bounds[0], room_bounds[1]
        self.y_min, self.y_max = room_bounds[2], room_bounds[3]
        self.z_min, self.z_max = room_bounds[4], room_bounds[5]
        
        self.min_kpts = min_keypoints_in_roi
        self.max_height = max_skeleton_height_m
        self.min_height = min_skeleton_height_m
        self.exclusion_zones: List[Tuple[float, float, float, float, float, float]] = []

    def add_exclusion_zone(self, bounds: Tuple[float, float, float, float, float, float]):
        """Adds a 3D box where RF noise (e.g. ceiling fans) should be ignored."""
        self.exclusion_zones.append(bounds)

    def is_point_inside_roi(self, x: float, y: float, z: float) -> bool:
        if not (self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max and self.z_min <= z <= self.z_max):
            return False
        for (ex_min, ex_max, ey_min, ey_max, ez_min, ez_max) in self.exclusion_zones:
            if (ex_min <= x <= ex_max and ey_min <= y <= ey_max and ez_min <= z <= ez_max):
                return False
        return True

    def filter_skeleton_3d(self, skeleton_3d: np.ndarray) -> Optional[np.ndarray]:
        """
        Validates 17-joint 3D skeleton. Returns sanitized skeleton or None if ghost/out-of-bounds.
        Args:
            skeleton_3d: [17, 3] array
        """
        if skeleton_3d is None or len(skeleton_3d) != 17:
            return None

        # 1. Check ROI Keypoint Containment
        valid_kpts_count = 0
        for joint in skeleton_3d:
            if self.is_point_inside_roi(joint[0], joint[1], joint[2]):
                valid_kpts_count += 1

        if valid_kpts_count < self.min_kpts:
            return None  # Out of room boundary

        # 2. Check Anatomical Height Plausibility (Head/Nose to Ankles)
        # Nose: Joint 0, Left Ankle: Joint 15, Right Ankle: Joint 16
        head_z = skeleton_3d[0, 2]
        ankle_z = 0.5 * (skeleton_3d[15, 2] + skeleton_3d[16, 2])
        estimated_height = abs(head_z - ankle_z)

        if not (self.min_height <= estimated_height <= self.max_height):
            # Out-of-bounds height likely indicates multipath stretching
            return None

        return skeleton_3d

    def compute_3d_bounding_box(self, skeleton_3d: np.ndarray) -> Dict[str, List[float]]:
        """
        Calculates 3D Axis-Aligned Bounding Box (AABB) with safety margins.
        """
        min_coords = np.min(skeleton_3d, axis=0) - 0.1  # 10cm padding
        max_coords = np.max(skeleton_3d, axis=0) + 0.1

        return {
            "center": ((min_coords + max_coords) * 0.5).tolist(),
            "dimensions": (max_coords - min_coords).tolist(),
            "min_bounds": min_coords.tolist(),
            "max_bounds": max_coords.tolist()
        }
