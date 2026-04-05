import numpy as np

ELBOW_RANGE = (60, 140)
ELBOW_SERVO_RANGE = (40, 140)
SHOULDER_DIRECTION_RANGE = (-90, 90)
SHOULDER_SERVO_RANGE = (20, 160)
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


class GestureModelVector:
    def __init__(self):
        self.prev_s2 = DEFAULT_SERVO_ANGLES[1]
        self.prev_s3 = DEFAULT_SERVO_ANGLES[2]
        self.prev_s4 = 90
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

    def compute_torso_offset(self, left_hip, right_hip):
        if not (is_valid_point(left_hip) and is_valid_point(right_hip)):
            raise ValueError("Invalid torso center inputs")

        center_x = (left_hip[0] + right_hip[0]) / 2
        if not is_finite_number(center_x):
            raise ValueError("Invalid torso center position")

        offset = center_x - 0.5
        offset *= 2

        if abs(offset) < 0.05:
            offset = 0.0

        return self.safe_number(offset, 0.0)

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

    def map_range(self, value, source_range, target_range, fallback):
        fallback = self.safe_number(fallback, 0.0)
        if not is_finite_number(value):
            return fallback

        source_min, source_max = source_range
        target_min, target_max = target_range
        clipped = float(np.clip(float(value), source_min, source_max))
        mapped = np.interp(clipped, [source_min, source_max], [target_min, target_max])
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

            shoulder, elbow, wrist, _l_sh, _r_sh, l_hip, r_hip = pose

            elbow_angle = calculate_angle(shoulder, elbow, wrist)

            if all(is_valid_point(point) for point in (shoulder, elbow)):
                vec_x = elbow[0] - shoulder[0]
                vec_y = elbow[1] - shoulder[1]
                mag = np.sqrt(vec_x**2 + vec_y**2)
                if mag != 0:
                    vec_x /= mag
                    vec_y /= mag

                shoulder_angle = np.degrees(np.arctan2(-vec_y, vec_x))
                vertical = elbow[1] - shoulder[1]
                combined = 0.7 * shoulder_angle + 0.3 * (vertical * 180)
            else:
                combined = np.nan

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

            if is_finite_number(elbow_angle):
                s2 = np.interp(elbow_angle, ELBOW_RANGE, ELBOW_SERVO_RANGE)
                s2 = self.safe_number(np.clip(s2, *ELBOW_SERVO_RANGE), previous_output[1])
            else:
                s2 = previous_output[1]

            if is_finite_number(combined):
                s3 = np.interp(combined, SHOULDER_DIRECTION_RANGE, SHOULDER_SERVO_RANGE)
                s3 = self.safe_number(np.clip(s3, *SHOULDER_SERVO_RANGE), previous_output[2])
            else:
                s3 = previous_output[2]

            alpha = 0.4
            s2 = alpha * s2 + (1 - alpha) * self.prev_s2
            s3 = alpha * s3 + (1 - alpha) * self.prev_s3

            s2 = int(round(np.clip(self.safe_number(s2, previous_output[1]), *ELBOW_SERVO_RANGE)))
            s3 = int(round(np.clip(self.safe_number(s3, previous_output[2]), *SHOULDER_SERVO_RANGE)))

            s2 = int(np.clip(s2, *ELBOW_SERVO_RANGE))
            s3 = int(np.clip(s3, *SHOULDER_SERVO_RANGE))

            try:
                offset = self.compute_torso_offset(l_hip, r_hip)
                if not is_finite_number(offset):
                    raise ValueError("Computed torso offset is not finite")

                s4 = np.interp(offset, [-1, 1], [0, 180])
                s4 = int(np.clip(s4, 0, 180))

                alpha = 0.3
                s4 = int(alpha * s4 + (1 - alpha) * self.prev_s4)

                max_step = 3
                delta = s4 - self.prev_s4

                if abs(delta) > max_step:
                    s4 = int(self.prev_s4 + max_step * np.sign(delta))
            except Exception:
                s4 = int(self.prev_s4)

            s1 = 180 if grip else 0
            output = (
                int(np.clip(self.safe_number(s1, previous_output[0]), 0, 180)),
                int(np.clip(self.safe_number(s2, previous_output[1]), *ELBOW_SERVO_RANGE)),
                int(np.clip(self.safe_number(s3, previous_output[2]), 20, 160)),
                int(np.clip(self.safe_number(s4, previous_output[3]), 0, 180)),
            )

            self.prev_s2 = output[1]
            self.prev_s3 = output[2]
            self.prev_s4 = output[3]
            self.prev_grip = grip
            self.prev_output = output
            return self.debug_output(output)
        except Exception:
            self.prev_output = previous_output
            return self.debug_output(previous_output)
