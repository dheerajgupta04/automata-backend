import cv2
import numpy as np
from scipy.signal import butter, filtfilt


# -------------------------
# Bandpass Filter
# -------------------------
def bandpass_filter(signal, fps, low=0.7, high=3.0, order=3):
    nyquist = 0.5 * fps
    low = low / nyquist
    high = high / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def calculate_sqi(rgb_signal):
    """
    Signal Quality Index: Checks for signal stability.
    """
    if len(rgb_signal) < 10:
        return 0.0
    
    rgb = np.array(rgb_signal)
    from scipy.signal import detrend
    detrended = detrend(rgb, axis=0)
    
    std_rgb = np.std(detrended, axis=0)
    mean_rgb = np.mean(rgb, axis=0)
    
    cv = std_rgb / (mean_rgb + 1e-6)
    if np.any(cv > 0.08): 
        return 0.3
    return 1.0


def extract_chrom_signal(rgb_signal):
    """
    Chrominance-based rPPG (CHROM).
    """
    X = rgb_signal[:, 0]
    Y = rgb_signal[:, 1]
    Z = rgb_signal[:, 2]
    
    X_n = X / (np.mean(X) + 1e-6)
    Y_n = Y / (np.mean(Y) + 1e-6)
    Z_n = Z / (np.mean(Z) + 1e-6)
    
    Xs = 3 * X_n - 2 * Y_n
    Ys = 1.5 * X_n + Y_n - 1.5 * Z_n
    
    alpha = np.std(Xs) / (np.std(Ys) + 1e-6)
    chrom = Xs - alpha * Ys
    return chrom


def extract_rgb_means(rois):
    """
    ROI-level quality gating: Calculates quality for each ROI before merging.
    """
    if not rois:
        return None

    valid_rois_data = []
    roi_scores = []

    # Skin mask bounds
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([25, 180, 255], dtype=np.uint8)

    for roi in rois:
        if roi is None or roi.size == 0: continue
        
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, lower, upper)
        
        skin_pixels_bgr = roi[mask > 0]
        if len(skin_pixels_bgr) < 50: continue
        
        # Quality Metric: Green-channel relative variance (Inverse)
        g_channel = skin_pixels_bgr[:, 1]
        cv = np.std(g_channel) / (np.mean(g_channel) + 1e-6)
        roi_scores.append(1.0 / (cv + 0.01))
        valid_rois_data.append(skin_pixels_bgr)

    if not valid_rois_data: return None

    # Pick top ROIs
    top_indices = np.argsort(roi_scores)[::-1]
    selected = [valid_rois_data[top_indices[0]]]
    if len(top_indices) > 1 and roi_scores[top_indices[0]] < 1.3 * roi_scores[top_indices[1]]:
        selected.append(valid_rois_data[top_indices[1]])

    merged = np.concatenate(selected, axis=0)
    mean_bgr = np.mean(merged, axis=0)
    
    # Return as [R, G, B]
    return [float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])]