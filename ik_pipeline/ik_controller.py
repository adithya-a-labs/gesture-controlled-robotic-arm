from ik_pipeline.semi_ik import solve_ik


class IKController:
    def __init__(self, L1=3.0, L2=3.0):
        self.L1 = L1
        self.L2 = L2

    def compute(self, shoulder_pt, wrist_pt):
        x = wrist_pt[0] - shoulder_pt[0]
        y = shoulder_pt[1] - wrist_pt[1]

        theta1, theta2 = solve_ik(x, y, self.L1, self.L2)

        return {
            "shoulder_angle": float(theta1),
            "elbow_angle": float(theta2)
        }
