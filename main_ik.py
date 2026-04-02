"""Main entrypoint for IK-based gesture controlled robotic arm."""

from __future__ import annotations

import cv2

from ik_pipeline.controller import ArmController
from ik_pipeline.ik_model import solve_ik
from ik_pipeline.tracker import PoseTracker


def main():
    cap = cv2.VideoCapture(0)
    tracker = PoseTracker()
    controller = ArmController()

    L1 = 0.3
    L2 = 0.3

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        tracked = tracker.process(frame)
        if tracked is not None:
            shoulder, elbow, wrist = tracked

            theta1, theta2 = solve_ik(shoulder, wrist, L1, L2)
            s3, s2 = controller.map_to_servo(theta1, theta2)

            cv2.putText(frame, f"theta1: {theta1:.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"theta2: {theta2:.1f} deg", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"s3: {s3:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"s2: {s2:.1f}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Optional visual markers for debugging.
            h, w = frame.shape[:2]
            for p, color in ((shoulder, (255, 0, 0)), (elbow, (0, 255, 0)), (wrist, (0, 0, 255))):
                px = int(p[0] * w)
                py = int(p[1] * h)
                cv2.circle(frame, (px, py), 5, color, -1)

        cv2.imshow("IK Arm Control", frame)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
