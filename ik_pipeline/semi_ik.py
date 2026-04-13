import numpy as np


def solve_ik(x, y, L1=3.0, L2=3.0):
    dist = np.sqrt(x**2 + y**2)
    dist = np.clip(dist, 0.001, L1 + L2 - 0.001)

    cos_theta2 = (dist**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)

    theta2 = np.arccos(cos_theta2)

    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)

    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)

    return np.degrees(theta1), np.degrees(theta2)
