"""Servo mapping controller for IK angles."""

from __future__ import annotations

from ik_pipeline.utils import clamp


class ArmController:
    """Map IK joint angles to shoulder/elbow servo commands."""

    def map_to_servo(self, theta1: float, theta2: float):
        """Map IK angles (deg) to 0-180 linear servo range.

        Returns:
            tuple[float, float]: (s3_shoulder, s2_elbow)
        """
        # Simple linear mapping for now.
        s3 = clamp(theta1 + 90.0, 0.0, 180.0)
        s2 = clamp(theta2, 0.0, 180.0)
        return s3, s2
