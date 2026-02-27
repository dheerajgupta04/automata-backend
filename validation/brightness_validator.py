import numpy as np
import cv2
from config import *

def validate_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    std = np.std(gray)
    contrast = gray.max() - gray.min()

    green_mean = np.mean(frame[:,:,1])

    result = {
        "mean": float(mean),
        "std": float(std),
        "green_mean": float(green_mean),
        "contrast": float(contrast),
        "valid": True
    }

    if mean < DARK_THRESHOLD or mean > BRIGHT_THRESHOLD:
        result["valid"] = False
    if not (GREEN_LOW <= green_mean <= GREEN_HIGH):
        result["valid"] = False
    if contrast < MIN_CONTRAST:
        result["valid"] = False

    return result