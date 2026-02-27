import numpy as np

def extract_roi(frame, landmarks):

    h, w, _ = frame.shape

    # Stable FaceMesh landmark indices
    forehead = landmarks.landmark[10]
    left_cheek = landmarks.landmark[234]
    right_cheek = landmarks.landmark[454]

    coords = []

    for lm in [forehead, left_cheek, right_cheek]:
        px = int(lm.x * w)
        py = int(lm.y * h)
        coords.append((px, py))

    x_vals = [c[0] for c in coords]
    y_vals = [c[1] for c in coords]

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    roi = frame[y_min:y_max, x_min:x_max]

    return roi