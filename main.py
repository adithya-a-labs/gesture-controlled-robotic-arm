"""
DEV MODE:

- USE_SERIAL = False -> runs without ESP32 (testing, debugging)
- USE_SERIAL = True  -> sends data to ESP32

This allows working on vision + dashboard without hardware connected.

SERVO MAPPING (IMPORTANT)

S4 -> Base rotation (torso left/right)
S3 -> Shoulder joint (arm up/down)
S2 -> Elbow joint (arm bending)
S1 -> Gripper (pinch open/close)

DATA FORMAT SENT TO ESP32:
    s4, s3, s2, s1

Example:
    90, 60, 120, 0

Which means:
    Base facing forward
    Shoulder slightly raised
    Elbow bent
    Gripper open
"""

import threading
import time

import cv2
import serial
from dashboard.server import update_state, socketio, app
from vision.camera import Camera
from vision.handtracking import HolisticTracker
from vision.gesture_model import GestureModel

USE_SERIAL = True  # Set True when ESP32 is connected

latest_frame = None
pose_result = None
hand_result = None
servo_angles = None
running = True
ser = None


def camera_loop(cam):
    global latest_frame, running

    while running:
        frame = cam.get_frame()
        if frame is not None:
            latest_frame = frame


def processing_loop(tracker, model):
    global latest_frame, pose_result, hand_result, servo_angles, running, ser

    while running:
        if latest_frame is None:
            continue

        frame = latest_frame.copy()

        p_result, h_result = tracker.process(frame)
        angles = model.compute_servo_angles(p_result, h_result)

        pose_result = p_result
        hand_result = h_result

        if angles:
            servo_angles = angles
            s1, s2, s3, s4 = servo_angles
            update_state(s1, s2, s3, s4)

            # ESP32 expects values in base, shoulder, elbow, gripper order.
            data = f"{s4},{s3},{s2},{s1}\n"

            if ser:
                try:
                    ser.write(data.encode())
                    print(f"Sent: {s4},{s3},{s2},{s1}")
                except Exception as e:
                    print("Serial write error:", e)

            if not ser:
                print(f"[DEV MODE] {data.strip()}")


def run_server():
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)


cam = Camera()
tracker = HolisticTracker()
model = GestureModel()

ser = None

if USE_SERIAL:
    try:
        ser = serial.Serial('COM5', 115200, timeout=1)
        print("Serial connected on COM5")
        time.sleep(2)  # allow ESP32 to initialize
    except Exception as e:
        print("Serial not available, running without hardware:", e)
        ser = None

threading.Thread(target=run_server, daemon=True).start()

cam_thread = threading.Thread(target=camera_loop, args=(cam,))
proc_thread = threading.Thread(target=processing_loop, args=(tracker, model))

cam_thread.start()
proc_thread.start()

try:
    while True:
        if latest_frame is None:
            continue

        frame = latest_frame.copy()

        if pose_result is not None and hand_result is not None:
            frame = tracker.draw(frame, pose_result, hand_result)

        if servo_angles:
            s1, s2, s3, s4 = servo_angles

            cv2.putText(frame, f"S4 Base: {int(s4)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"S3 Shoulder: {int(s3)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"S2 Elbow: {int(s2)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"S1 Gripper: {int(s1)}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Arm Control", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            running = False
            break
finally:
    running = False
    cam.release()
    cv2.destroyAllWindows()
    cam_thread.join()
    proc_thread.join()
    if ser:
        ser.close()
