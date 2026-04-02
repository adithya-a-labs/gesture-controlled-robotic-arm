"""Utility helpers for IK pipeline."""

from __future__ import annotations

import numpy as np


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp *value* inside the inclusive [min_val, max_val] range."""
    return max(min_val, min(value, max_val))


def distance(p1, p2) -> float:
    """Compute 2D Euclidean distance between two points."""
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    return float(np.linalg.norm(p2 - p1))
