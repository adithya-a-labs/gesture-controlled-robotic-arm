import cv2
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


class HolisticTracker:
    def __init__(self):
        # Pose model
        pose_base = python.BaseOptions(model_asset_path="pose_landmarker.task")
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base,
            running_mode=vision.RunningMode.VIDEO
        )
        self.pose = vision.PoseLandmarker.create_from_options(pose_options)

        # Hand model
        hand_base = python.BaseOptions(model_asset_path="hand_landmarker.task")
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base,
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO
        )
        self.hand = vision.HandLandmarker.create_from_options(hand_options)

        self.timestamp = 0

    def process(self, frame):
        self.timestamp += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        pose_result = self.pose.detect_for_video(mp_image, self.timestamp)
        hand_result = self.hand.detect_for_video(mp_image, self.timestamp)

        return pose_result, hand_result

    def draw(self, frame, pose_result, hand_result):
        h, w, _ = frame.shape

        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]

            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            def get_point(i):
                return int(landmarks[i].x * w), int(landmarks[i].y * h)

            shoulder = get_point(12)
            elbow = get_point(14)
            wrist = get_point(16)

            cv2.line(frame, shoulder, elbow, (255, 0, 0), 3)
            cv2.line(frame, elbow, wrist, (255, 0, 0), 3)

        if hand_result.hand_landmarks:
            landmarks = hand_result.hand_landmarks[0]

            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        return frame
    
