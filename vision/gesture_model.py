from collections import deque

import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    angle = np.arccos(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    )

    return np.degrees(angle)


class GestureModel:
    def __init__(self):
        self.s2_buffer = deque(maxlen=5)
        self.s3_buffer = deque(maxlen=5)

    def smooth(self, buffer, value):
        buffer.append(value)
        return int(sum(buffer) / len(buffer))

    def get_pose_points(self, pose_result):
        if not pose_result.pose_landmarks:
            return None

        lm = pose_result.pose_landmarks[0]

        if any(lm[i].visibility < 0.5 for i in (12, 14, 16)):
            return None

        shoulder = [lm[12].x, lm[12].y]
        elbow = [lm[14].x, lm[14].y]
        wrist = [lm[16].x, lm[16].y]

        return shoulder, elbow, wrist

    def get_hand_points(self, hand_result):
        if not hand_result.hand_landmarks:
            return None

        lm = hand_result.hand_landmarks[0]

        thumb_tip = [lm[4].x, lm[4].y]
        index_tip = [lm[8].x, lm[8].y]
        index_mcp = [lm[5].x, lm[5].y]
        wrist = [lm[0].x, lm[0].y]

        return thumb_tip, index_tip, index_mcp, wrist

    def compute_servo_angles(self, pose_result, hand_result):
        pose = self.get_pose_points(pose_result)
        hand = self.get_hand_points(hand_result)

        if pose is None:
            return None

        shoulder, elbow, wrist = pose

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        vertical = [shoulder[0], shoulder[1] - 0.2]
        shoulder_angle = calculate_angle(elbow, shoulder, vertical)

        grip = 0
        if hand:
            thumb_tip, index_tip, index_mcp, hand_wrist = hand
            tip_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
            ref_dist = np.linalg.norm(np.array(index_mcp) - np.array(hand_wrist))

            if ref_dist > 0:
                ratio = tip_dist / ref_dist
                grip = 1 if ratio < 0.3 else 0

        s2 = int(np.interp(elbow_angle, [30, 160], [0, 180]))
        s3 = int(np.interp(shoulder_angle, [20, 150], [0, 180]))
        s2 = self.smooth(self.s2_buffer, s2)
        s3 = self.smooth(self.s3_buffer, s3)
        s1 = 180 if grip else 0

        return s1, s2, s3
