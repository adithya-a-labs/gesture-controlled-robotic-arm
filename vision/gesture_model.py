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
        pass

    def get_pose_points(self, pose_result):
        if not pose_result.pose_landmarks:
            return None

        lm = pose_result.pose_landmarks[0]

        shoulder = [lm[12].x, lm[12].y]
        elbow = [lm[14].x, lm[14].y]
        wrist = [lm[16].x, lm[16].y]

        return shoulder, elbow, wrist


    def get_hand_points(self, hand_result):
        if not hand_result.hand_landmarks:
            return None

        lm = hand_result.hand_landmarks[0]

        thumb = [lm[4].x, lm[4].y]
        index = [lm[8].x, lm[8].y]

        return thumb, index
    def compute_servo_angles(self, pose_result, hand_result):
        pose = self.get_pose_points(pose_result)
        hand = self.get_hand_points(hand_result)

        if pose is None:
            return None

        shoulder, elbow, wrist = pose

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        shoulder_angle = calculate_angle(elbow, shoulder, [shoulder[0], 0])

        grip = 0
        if hand:
            thumb, index = hand
            dist = np.linalg.norm(np.array(thumb) - np.array(index))
            grip = 1 if dist < 0.05 else 0

        s2 = int(np.interp(elbow_angle, [30, 160], [0, 180]))
        s3 = int(np.interp(shoulder_angle, [20, 150], [0, 180]))
        s1 = 180 if grip else 0

        return s1, s2, s3