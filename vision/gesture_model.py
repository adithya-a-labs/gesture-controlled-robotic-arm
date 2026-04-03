import numpy as np

ELBOW_RANGE = (20, 150)
SHOULDER_RANGE = (10, 160)
TORSO_RANGE = (-45, 45)
DEFAULT_SERVO_ANGLES = (90, 90, 90, 90)


def is_finite_number(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def is_valid_point(point):
    try:
        return (
            point is not None
            and len(point) == 2
            and all(is_finite_number(coord) for coord in point)
        )
    except TypeError:
        return False


def calculate_angle(a, b, c):
    if not all(is_valid_point(point) for point in (a, b, c)):
        return np.nan

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator <= 0 or not np.isfinite(denominator):
        return np.nan

    cosine = np.dot(ba, bc) / denominator
    if not np.isfinite(cosine):
        return np.nan

    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


class GestureModel:
    def __init__(self):
        self.prev_s2 = DEFAULT_SERVO_ANGLES[1]
        self.prev_s3 = DEFAULT_SERVO_ANGLES[2]
        self.prev_s4 = 90
        self.center_offset = None
        self.prev_grip = 0
        self.prev_pose_points = None
        self.prev_output = DEFAULT_SERVO_ANGLES

    def safe_number(self, value, fallback=0.0):
        if is_finite_number(value):
            return float(value)
        if is_finite_number(fallback):
            return float(fallback)
        return 0.0

    def smooth(self, current, prev, alpha):
        current = self.safe_number(current, prev)
        if prev is None or not is_finite_number(prev):
            return current

        alpha = self.safe_number(alpha, 0.3)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return alpha * current + (1 - alpha) * float(prev)

    def limit_speed(self, target, current, max_step=5):
        target = self.safe_number(target, current)
        if current is None or not is_finite_number(current):
            return target

        current = float(current)
        delta = target - current
        if abs(delta) > max_step:
            target = current + max_step * np.sign(delta)

        return target

    def get_adaptive_alpha(self, current, prev):
        if prev is None or not is_finite_number(prev):
            return 0.7

        current = self.safe_number(current, prev)
        velocity = abs(current - float(prev))
        return 0.7 if velocity > 10 else 0.3

    def compute_angle(self, p1, p2):
        if not (is_valid_point(p1) and is_valid_point(p2)):
            return np.nan

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        if not (is_finite_number(dx) and is_finite_number(dy)):
            return np.nan

        return float(np.degrees(np.arctan2(dy, dx)))

    def compute_torso_angle(self, left_shoulder, right_shoulder, left_hip, right_hip):
        hip_angle = self.compute_angle(left_hip, right_hip)
        shoulder_angle = self.compute_angle(left_shoulder, right_shoulder)

        if not (is_finite_number(hip_angle) and is_finite_number(shoulder_angle)):
            raise ValueError("Invalid torso angle inputs")

        torso_angle = 0.7 * hip_angle + 0.3 * shoulder_angle

        if self.center_offset is None:
            self.center_offset = torso_angle

        torso_angle = torso_angle - self.center_offset

        if abs(torso_angle) < 5:
            torso_angle = 0.0

        torso_angle = -torso_angle
        torso_angle *= 1.5

        return self.safe_number(torso_angle, 0.0)

    def landmark_to_point(self, landmark):
        x = getattr(landmark, "x", None)
        y = getattr(landmark, "y", None)

        if not (is_finite_number(x) and is_finite_number(y)):
            return []

        point = [float(x), float(y)]
        return point if is_valid_point(point) else []

    def resolve_pose_point(self, landmarks, index, previous_point):
        if index >= len(landmarks):
            return previous_point

        landmark = landmarks[index]
        visibility = getattr(landmark, "visibility", 1.0)
        point = self.landmark_to_point(landmark)

        if not is_valid_point(point) or not is_finite_number(visibility) or visibility < 0.5:
            return previous_point

        return point

    def get_pose_points(self, pose_result):
        if not pose_result or not getattr(pose_result, "pose_landmarks", None):
            return self.prev_pose_points

        landmarks = pose_result.pose_landmarks[0]
        previous = self.prev_pose_points or (None,) * 7

        shoulder = self.resolve_pose_point(landmarks, 12, previous[0])
        elbow = self.resolve_pose_point(landmarks, 14, previous[1])
        wrist = self.resolve_pose_point(landmarks, 16, previous[2])
        left_shoulder = self.resolve_pose_point(landmarks, 11, previous[3])
        right_shoulder = self.resolve_pose_point(landmarks, 12, previous[4])
        left_hip = self.resolve_pose_point(landmarks, 23, previous[5])
        right_hip = self.resolve_pose_point(landmarks, 24, previous[6])

        points = (
            shoulder,
            elbow,
            wrist,
            left_shoulder,
            right_shoulder,
            left_hip,
            right_hip,
        )

        if any(point is None for point in points):
            return self.prev_pose_points

        self.prev_pose_points = points
        return points

    def get_hand_points(self, hand_result):
        if not hand_result or not getattr(hand_result, "hand_landmarks", None):
            return ()

        landmarks = hand_result.hand_landmarks[0]
        required_indexes = (4, 8, 5, 0)
        if len(landmarks) <= max(required_indexes):
            return ()

        points = tuple(self.landmark_to_point(landmarks[i]) for i in required_indexes)

        if any(not is_valid_point(point) for point in points):
            return ()

        return points

    def get_previous_output(self):
        return tuple(
            int(round(np.clip(self.safe_number(value, DEFAULT_SERVO_ANGLES[i]), 0, 180)))
            for i, value in enumerate(self.prev_output)
        )

    def map_angle_to_servo(self, angle, source_range, fallback):
        fallback = self.safe_number(fallback, 0.0)
        if not is_finite_number(angle):
            return fallback

        clipped = float(np.clip(float(angle), *source_range))
        mapped = np.interp(clipped, source_range, [0, 180])
        return self.safe_number(mapped, fallback)

    def finalize_servo(self, target, prev, fallback, max_step=5):
        limited = self.limit_speed(target, prev, max_step=max_step)
        value = self.safe_number(limited, fallback)
        return int(round(np.clip(value, 0, 180)))

    def debug_output(self, output):
        s1, s2, s3, s4 = output
        print("DEBUG:", s1, s2, s3, s4)
        return output

    def compute_servo_angles(self, pose_result, hand_result):
        previous_output = self.get_previous_output()
        try:
            pose = self.get_pose_points(pose_result)
            hand = self.get_hand_points(hand_result)

            if pose is None:
                self.prev_output = previous_output
                return self.debug_output(previous_output)

            shoulder, elbow, wrist, l_sh, r_sh, l_hip, r_hip = pose

            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            vertical = [shoulder[0], shoulder[1] - 0.2]
            shoulder_angle = calculate_angle(elbow, shoulder, vertical)

            grip = self.prev_grip
            if hand:
                thumb_tip, index_tip, index_mcp, hand_wrist = hand
                tip_dist = np.linalg.norm(np.array(thumb_tip, dtype=float) - np.array(index_tip, dtype=float))
                ref_dist = np.linalg.norm(np.array(index_mcp, dtype=float) - np.array(hand_wrist, dtype=float))

                if is_finite_number(tip_dist) and is_finite_number(ref_dist) and ref_dist > 0:
                    ratio = tip_dist / ref_dist
                    if ratio < 0.25:
                        grip = 1
                    elif ratio > 0.35:
                        grip = 0

            raw_s2 = self.map_angle_to_servo(elbow_angle, ELBOW_RANGE, previous_output[1])
            raw_s3 = self.map_angle_to_servo(shoulder_angle, SHOULDER_RANGE, previous_output[2])

            s2_alpha = self.get_adaptive_alpha(raw_s2, self.prev_s2)
            s3_alpha = self.get_adaptive_alpha(raw_s3, self.prev_s3)

            smooth_s2 = self.smooth(raw_s2, self.prev_s2, s2_alpha)
            smooth_s3 = self.smooth(raw_s3, self.prev_s3, s3_alpha)

            s2 = self.finalize_servo(smooth_s2, self.prev_s2, previous_output[1])
            s3 = self.finalize_servo(smooth_s3, self.prev_s3, previous_output[2])
            try:
                torso_angle = self.compute_torso_angle(l_sh, r_sh, l_hip, r_hip)
                if not is_finite_number(torso_angle):
                    raise ValueError("Computed torso angle is not finite")

                s4 = np.interp(torso_angle, [-90, 90], [0, 180])
                s4 = int(np.clip(s4, 0, 180))

                alpha = 0.4
                s4 = int(alpha * s4 + (1 - alpha) * self.prev_s4)

                max_step = 4
                delta = s4 - self.prev_s4

                if abs(delta) > max_step:
                    s4 = int(self.prev_s4 + max_step * np.sign(delta))
            except Exception:
                s4 = int(self.prev_s4)

            s1 = 180 if grip else 0

            if any(v is None for v in (s1, s2, s3, s4)):
                self.prev_output = previous_output
                return self.debug_output(previous_output)

            self.prev_s2 = s2
            self.prev_s3 = s3
            self.prev_s4 = s4
            self.prev_grip = grip
            self.prev_output = (s1, s2, s3, s4)
            return self.debug_output(self.prev_output)
        except Exception:
            self.prev_output = previous_output
            return self.debug_output(previous_output)
