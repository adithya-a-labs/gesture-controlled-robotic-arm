import threading

import cv2
from flask import render_template
from dashboard.server import update_state, socketio, app
from vision.camera import Camera
from vision.handtracking import HolisticTracker
from vision.gesture_model_vector import GestureModelVector

latest_frame = None
pose_result = None
hand_result = None
servo_angles = None
running = True


def camera_loop(cam):
    global latest_frame, running

    while running:
        frame = cam.get_frame()
        if frame is not None:
            latest_frame = frame


def processing_loop(tracker, model):
    global latest_frame, pose_result, hand_result, servo_angles, running

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


def run_server():
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)


@app.route("/3d-vector")
def dashboard3d_vector():
    return render_template("index3d_vector_fk.html")


cam = Camera()
tracker = HolisticTracker()
model = GestureModelVector()

threading.Thread(target=run_server, daemon=True).start()

cam_thread = threading.Thread(target=camera_loop, args=(cam,))
proc_thread = threading.Thread(target=processing_loop, args=(tracker, model))

cam_thread.start()
proc_thread.start()

while True:
    if latest_frame is None:
        continue

    frame = latest_frame.copy()

    if pose_result is not None and hand_result is not None:
        frame = tracker.draw(frame, pose_result, hand_result)

    if servo_angles:
        s1, s2, s3, s4 = servo_angles

        cv2.putText(frame, f"S1: {s1}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"S2: {s2}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"S3: {s3}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"S4 (Base): {s4}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Arm Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        running = False
        break

cam.release()
cam_thread.join()
proc_thread.join()
