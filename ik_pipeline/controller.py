"""Servo mapping controller for IK angles."""

from __future__ import annotations

import numpy as np

from calibration import get_calibration, get_elbow_servo_range, get_shoulder_center, get_shoulder_servo_range


class ArmController:
    """Map IK joint angles to shoulder/elbow servo commands."""

    def map_to_servo(self, theta1: float, theta2: float):
        """Map IK angles (deg) to constrained shoulder/elbow servo commands.

        Returns:
            tuple[float, float]: (s3_shoulder, s2_elbow)
        """
        config = get_calibration()
        s2_min, s2_max = get_elbow_servo_range(config)
        s3_center = get_shoulder_center(config)
        _, s3_max = get_shoulder_servo_range(config)

        s3 = np.interp(theta1, [0.0, 90.0], [s3_max, s3_center])
        s3 = float(np.clip(s3, s3_center, s3_max))

        s2 = np.interp(theta2, [60.0, 180.0], [s2_min, s2_max])
        s2 = float(np.clip(s2, s2_min, s2_max))
        return s3, s2
