import cv2
from config import MIN_WIDTH, MIN_HEIGHT, MIN_DISTANCE_CM, MAX_DISTANCE_CM

def validate_resolution(frame):
    height, width = frame.shape[:2]

    result = {
        "width": width,
        "height": height,
        "min_width_required": MIN_WIDTH,
        "min_height_required": MIN_HEIGHT,
        "valid": True
    }

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        result["valid"] = False

    return result


def validate_camera_distance(frame, landmarks):
    """
    Estimate distance using inter-ocular distance ratio.
    This is approximate.
    """

    h, w = frame.shape[:2]

    left_eye = landmarks.landmark[33]
    right_eye = landmarks.landmark[263]

    x1, y1 = int(left_eye.x * w), int(left_eye.y * h)
    x2, y2 = int(right_eye.x * w), int(right_eye.y * h)

    pixel_distance = ((x2 - x1)**2 + (y2 - y1)**2)**0.5

    # Approximate mapping (you can calibrate this later)
    # Assume ideal eye distance at 35 cm ≈ 120 pixels (example baseline)
    estimated_distance = 4200 / (pixel_distance + 1e-6)

    result = {
        "estimated_distance_cm": float(estimated_distance),
        "min_distance_cm": MIN_DISTANCE_CM,
        "max_distance_cm": MAX_DISTANCE_CM,
        "valid": True
    }

    if estimated_distance < MIN_DISTANCE_CM or estimated_distance > MAX_DISTANCE_CM:
        result["valid"] = False

    return result