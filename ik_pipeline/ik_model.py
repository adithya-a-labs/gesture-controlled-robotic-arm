"""Inverse kinematics model for a planar 2-link arm."""

from __future__ import annotations

import numpy as np

from ik_pipeline.utils import clamp


def solve_ik(shoulder, wrist, L1: float, L2: float):
    """Solve 2-link IK from shoulder to wrist in normalized image coordinates.

    Returns:
        tuple[float, float]: (theta1_deg, theta2_deg)
    """
    shoulder = np.asarray(shoulder, dtype=np.float32)
    wrist = np.asarray(wrist, dtype=np.float32)

    # Relative vector from shoulder to wrist.
    dx = float(wrist[0] - shoulder[0])
    dy = float(wrist[1] - shoulder[1])

    # Distance from shoulder to wrist target.
    d = float(np.sqrt(dx * dx + dy * dy))

    # Keep d in reachable/non-degenerate range to avoid invalid acos and division.
    eps = 1e-6
    d = clamp(d, eps, L1 + L2 - eps)

    # Elbow angle.
    cos_theta2 = (d * d - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_theta2 = clamp(cos_theta2, -1.0, 1.0)
    theta2 = float(np.arccos(cos_theta2))

    # Shoulder angle.
    theta1 = float(np.arctan2(dy, dx) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2)))

    # Convert to degrees.
    theta1_deg = float(np.degrees(theta1))
    theta2_deg = float(np.degrees(theta2))

    return theta1_deg, theta2_deg
