import numpy as np

ELBOW_RANGE = (20, 150)
SHOULDER_RANGE = (10, 160)

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
        self.prev_s2 = None
        self.prev_s3 = None
        self.prev_s4 = None
        self.prev_grip = 0

    def smooth(self, current, prev, alpha):
        if prev is None:
            return current
        return alpha * current + (1 - alpha) * prev

    def limit_speed(self, target, current, max_step=5):
        if current is None:
            return target

        delta = target - current
        if abs(delta) > max_step:
            target = current + max_step * np.sign(delta)

        return target

    def get_adaptive_alpha(self, current, prev):
        if prev is None:
            return 0.7

        velocity = abs(current - prev)
        return 0.7 if velocity > 10 else 0.3

    def compute_torso_angle(self, left, right):
        dx = right[0] - left[0]
        dy = right[1] - left[1]
        angle = np.arctan2(dy, dx)
        return np.degrees(angle)

    def get_pose_points(self, pose_result):
        if not pose_result.pose_landmarks:
            return None

        lm = pose_result.pose_landmarks[0]

        if any(lm[i].visibility < 0.5 for i in (12, 14, 16)):
            return None

        shoulder = [lm[12].x, lm[12].y]
        elbow = [lm[14].x, lm[14].y]
        wrist = [lm[16].x, lm[16].y]
        left_shoulder = [lm[11].x, lm[11].y]
        right_shoulder = [lm[12].x, lm[12].y]

        return shoulder, elbow, wrist, left_shoulder, right_shoulder

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

        shoulder, elbow, wrist, left_shoulder, right_shoulder = pose

        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        vertical = [shoulder[0], shoulder[1] - 0.2]
        shoulder_angle = calculate_angle(elbow, shoulder, vertical)
        torso_angle = self.compute_torso_angle(left_shoulder, right_shoulder)

        grip = self.prev_grip
        if hand:
            thumb_tip, index_tip, index_mcp, hand_wrist = hand
            tip_dist = np.linalg.norm(np.array(thumb_tip) - np.array(index_tip))
            ref_dist = np.linalg.norm(np.array(index_mcp) - np.array(hand_wrist))

            if ref_dist > 0:
                ratio = tip_dist / ref_dist
                if ratio < 0.25:
                    grip = 1
                elif ratio > 0.35:
                    grip = 0

        elbow_angle = float(np.clip(elbow_angle, *ELBOW_RANGE))
        shoulder_angle = float(np.clip(shoulder_angle, *SHOULDER_RANGE))

        raw_s2 = float(np.interp(elbow_angle, ELBOW_RANGE, [0, 180]))
        raw_s3 = float(np.interp(shoulder_angle, SHOULDER_RANGE, [0, 180]))
        raw_s4 = float(np.clip(np.interp(torso_angle, [-45, 45], [0, 180]), 0, 180))

        s2_alpha = self.get_adaptive_alpha(raw_s2, self.prev_s2)
        s3_alpha = self.get_adaptive_alpha(raw_s3, self.prev_s3)
        s4_alpha = self.get_adaptive_alpha(raw_s4, self.prev_s4)

        smooth_s2 = self.smooth(raw_s2, self.prev_s2, s2_alpha)
        smooth_s3 = self.smooth(raw_s3, self.prev_s3, s3_alpha)
        smooth_s4 = self.smooth(raw_s4, self.prev_s4, s4_alpha)

        s2 = int(round(self.limit_speed(smooth_s2, self.prev_s2)))
        s3 = int(round(self.limit_speed(smooth_s3, self.prev_s3)))
        s4 = int(round(self.limit_speed(smooth_s4, self.prev_s4)))

        self.prev_s2 = s2
        self.prev_s3 = s3
        self.prev_s4 = s4
        self.prev_grip = grip
        s1 = 180 if grip else 0

        return s1, s2, s3, s4
