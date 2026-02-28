import numpy as np

def extract_roi(frame, landmarks):
    """
    Extracts multiple skin-based ROIs: Forehead, Left Cheek, Right Cheek.
    """
    h, w, _ = frame.shape

    def get_coords(indices):
        points = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            points.append((int(lm.x * w), int(lm.y * h)))
        return points

    # Landmark indices for specific regions
    # Forehead: middle section, avoiding eyebrows and hair
    forehead_indices = [67, 109, 10, 338, 297]
    # Left Cheek: avoiding eyes/nose
    l_cheek_indices = [118, 119, 100, 126, 209]
    # Right Cheek: avoiding eyes/nose
    r_cheek_indices = [347, 348, 329, 355, 429]

    rois = []
    for region_indices in [forehead_indices, l_cheek_indices, r_cheek_indices]:
        coords = get_coords(region_indices)
        x_vals = [c[0] for c in coords]
        y_vals = [c[1] for c in coords]
        
        # Add some padding/buffer to the bounding box
        x_min, x_max = max(0, min(x_vals)), min(w, max(x_vals))
        y_min, y_max = max(0, min(y_vals)), min(h, max(y_vals))
        
        if x_max > x_min and y_max > y_min:
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size > 0:
                rois.append(roi)

    return rois