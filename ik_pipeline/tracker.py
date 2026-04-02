"""Pose tracking using MediaPipe Tasks PoseLandmarker."""

from __future__ import annotations

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseTracker:
    """Tracks shoulder, elbow, and wrist landmarks from camera frames."""

    def __init__(self, model_path: str = "pose_landmarker_full.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.pose = vision.PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def process(self, frame):
        """Process a BGR frame and return normalized [x, y] for shoulder/elbow/wrist."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use mediapipe.Image as requested.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.timestamp_ms += 33
        result = self.pose.detect_for_video(mp_image, self.timestamp_ms)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]

        shoulder = [landmarks[12].x, landmarks[12].y]
        elbow = [landmarks[14].x, landmarks[14].y]
        wrist = [landmarks[16].x, landmarks[16].y]

        return shoulder, elbow, wrist
