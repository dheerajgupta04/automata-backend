# # import numpy as np
# # from scipy.signal import butter, filtfilt

# # def bandpass_filter(signal, fps):
# #     low = 0.7
# #     high = 4.0
# #     nyq = 0.5 * fps
# #     b, a = butter(3, [low/nyq, high/nyq], btype='band')
# #     return filtfilt(b, a, signal)

# # def extract_green_signal(roi):
# #     return np.mean(roi[:,:,1])
# import numpy as np
# from scipy.signal import butter, filtfilt


# def bandpass_filter(signal, fps):
#     low = 0.7
#     high = 4.0
#     nyq = 0.5 * fps
#     b, a = butter(3, [low/nyq, high/nyq], btype='band')
#     return filtfilt(b, a, signal)


# def extract_rgb_means(roi):
#     """
#     Returns mean R, G, B values from ROI
#     """
#     r = np.mean(roi[:, :, 2])
#     g = np.mean(roi[:, :, 1])
#     b = np.mean(roi[:, :, 0])

#     return np.array([r, g, b])
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


# -------------------------
# Skin-Masked RGB Extraction
# -------------------------
def extract_rgb_means(roi):
    """
    Returns [R, G, B] values from skin-only pixels.
    Keeps original function name so batch file works.
    """

    if roi is None or roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Skin color range (tuned for indoor light)
    lower = np.array([0, 35, 50], dtype=np.uint8)
    upper = np.array([25, 200, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Clean small noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    skin_pixels = roi[mask > 0]

    if len(skin_pixels) < 100:
        return None

    mean_bgr = np.mean(skin_pixels, axis=0)

    r = mean_bgr[2]
    g = mean_bgr[1]
    b = mean_bgr[0]

    return [r, g, b]