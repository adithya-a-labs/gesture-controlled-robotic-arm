from __future__ import annotations

from threading import RLock

DEFAULT_CALIBRATION = {
    "s2_hmin": 60,
    "s2_hmax": 180,
    "s2_smin": 20,
    "s2_smax": 150,
    "s3_center": 55,
    "s3_min": 55,
    "s3_max": 100,
    "s4_center": 90,
    "s4_range": 60,
    "pinch_threshold": 0.04,
    "release_threshold": 0.07,
}

CALIBRATION_LIMITS = {
    "s2_hmin": (0, 180),
    "s2_hmax": (0, 180),
    "s2_smin": (0, 180),
    "s2_smax": (0, 180),
    "s3_center": (0, 180),
    "s3_min": (0, 180),
    "s3_max": (0, 180),
    "s4_center": (0, 180),
    "s4_range": (2, 180),
    "pinch_threshold": (0.0, 1.0),
    "release_threshold": (0.0, 1.0),
}

FLOAT_KEYS = {"pinch_threshold", "release_threshold"}
MIN_RANGE_GAP = 1
LEGACY_CALIBRATION_ALIASES = {
    "s3_offset": "s3_center",
}

_lock = RLock()

calibration = DEFAULT_CALIBRATION.copy()
saved_calibration = DEFAULT_CALIBRATION.copy()


def _clamp(value, low, high):
    return max(low, min(high, value))


def _coerce_value(key, value):
    default_value = DEFAULT_CALIBRATION[key]
    lower, upper = CALIBRATION_LIMITS[key]

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float(default_value)

    number = _clamp(number, lower, upper)

    if key in FLOAT_KEYS:
        return round(number, 4)

    return int(round(number))


def _normalize_int_range(lower, upper, absolute_min=0, absolute_max=180, minimum_gap=MIN_RANGE_GAP):
    lower = int(round(_clamp(float(lower), absolute_min, absolute_max)))
    upper = int(round(_clamp(float(upper), absolute_min, absolute_max)))

    if lower > upper:
        lower, upper = upper, lower

    if upper - lower < minimum_gap:
        if upper + minimum_gap <= absolute_max:
            upper += minimum_gap
        else:
            lower = max(absolute_min, upper - minimum_gap)

    return lower, upper


def sanitize_calibration(values=None):
    merged = DEFAULT_CALIBRATION.copy()
    if values:
        aliased_values = dict(values)
        for legacy_key, current_key in LEGACY_CALIBRATION_ALIASES.items():
            if current_key not in aliased_values and legacy_key in aliased_values:
                aliased_values[current_key] = aliased_values[legacy_key]

        merged.update(aliased_values)

    normalized = {
        key: _coerce_value(key, merged.get(key))
        for key in DEFAULT_CALIBRATION
    }

    normalized["s2_hmin"], normalized["s2_hmax"] = _normalize_int_range(
        normalized["s2_hmin"],
        normalized["s2_hmax"],
    )
    normalized["s2_smin"], normalized["s2_smax"] = _normalize_int_range(
        normalized["s2_smin"],
        normalized["s2_smax"],
    )
    normalized["s3_center"], normalized["s3_max"] = _normalize_int_range(
        normalized["s3_center"],
        normalized["s3_max"],
    )
    normalized["s3_min"] = normalized["s3_center"]

    pinch_threshold = float(normalized["pinch_threshold"])
    release_threshold = float(normalized["release_threshold"])
    if release_threshold < pinch_threshold:
        release_threshold = pinch_threshold

    normalized["pinch_threshold"] = round(pinch_threshold, 4)
    normalized["release_threshold"] = round(release_threshold, 4)
    return normalized


def get_calibration():
    with _lock:
        return calibration.copy()


def get_saved_calibration():
    with _lock:
        return saved_calibration.copy()


def update_calibration_values(values):
    with _lock:
        merged = calibration.copy()
        if values:
            merged.update(values)

        calibration.clear()
        calibration.update(sanitize_calibration(merged))
        return calibration.copy()


def save_calibration_values():
    with _lock:
        saved_calibration.clear()
        saved_calibration.update(calibration)
        return saved_calibration.copy()


def reset_calibration_values():
    with _lock:
        calibration.clear()
        calibration.update(DEFAULT_CALIBRATION)
        return calibration.copy()


def restore_calibration_values():
    with _lock:
        calibration.clear()
        calibration.update(saved_calibration)
        return calibration.copy()


def get_elbow_human_range(config=None):
    config = config or get_calibration()
    return _normalize_int_range(
        config.get("s2_hmin", DEFAULT_CALIBRATION["s2_hmin"]),
        config.get("s2_hmax", DEFAULT_CALIBRATION["s2_hmax"]),
    )


def get_elbow_servo_range(config=None):
    config = config or get_calibration()
    return _normalize_int_range(
        config.get("s2_smin", DEFAULT_CALIBRATION["s2_smin"]),
        config.get("s2_smax", DEFAULT_CALIBRATION["s2_smax"]),
    )


def get_shoulder_servo_range(config=None):
    config = config or get_calibration()
    return _normalize_int_range(
        config.get("s3_min", DEFAULT_CALIBRATION["s3_min"]),
        config.get("s3_max", DEFAULT_CALIBRATION["s3_max"]),
    )


def get_shoulder_center(config=None):
    config = config or get_calibration()
    return int(
        round(
            _clamp(
                config.get("s3_center", DEFAULT_CALIBRATION["s3_center"]),
                config.get("s3_min", DEFAULT_CALIBRATION["s3_min"]),
                config.get("s3_max", DEFAULT_CALIBRATION["s3_max"]),
            )
        )
    )


def get_base_servo_range(config=None):
    config = config or get_calibration()

    center = float(config.get("s4_center", DEFAULT_CALIBRATION["s4_center"]))
    servo_range = float(config.get("s4_range", DEFAULT_CALIBRATION["s4_range"]))
    servo_range = _clamp(servo_range, CALIBRATION_LIMITS["s4_range"][0], CALIBRATION_LIMITS["s4_range"][1])

    half_range = servo_range / 2.0
    minimum = _clamp(center - half_range, 0.0, 180.0)
    maximum = _clamp(center + half_range, 0.0, 180.0)

    if maximum - minimum < MIN_RANGE_GAP:
        maximum = _clamp(minimum + MIN_RANGE_GAP, 0.0, 180.0)
        minimum = _clamp(maximum - MIN_RANGE_GAP, 0.0, 180.0)

    return minimum, maximum
