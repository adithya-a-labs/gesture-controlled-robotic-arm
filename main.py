import cv2
from vision.camera import Camera
from vision.handtracking import HolisticTracker
from vision.gesture_model import GestureModel

cam = Camera()
tracker = HolisticTracker()
model = GestureModel()

while True:
    frame = cam.get_frame()
    if frame is None:
        break

    pose_result, hand_result = tracker.process(frame)

    servo_angles = model.compute_servo_angles(pose_result, hand_result)

    frame = tracker.draw(frame, pose_result, hand_result)

    if servo_angles:
        s1, s2, s3 = servo_angles

        cv2.putText(frame, f"S1 (Gripper): {s1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"S2 (Elbow): {s2}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"S3 (Shoulder): {s3}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("Arm Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
