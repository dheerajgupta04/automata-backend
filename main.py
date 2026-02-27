import cv2
import numpy as np
import joblib
from scipy.signal import find_peaks

from config import TOTAL_FRAMES
from validation.face_validator import FaceValidator
from processing.roi_extractor import extract_roi
from processing.signal_processor import extract_rgb_means
from processing.hr_estimation import estimate_hr, get_pos_signal
from processing.rr_estimation import estimate_rr
from processing.hrv_estimation import estimate_hrv
from processing.spo2_estimation import estimate_spo2


# =========================================================
# 🔥 LOAD BP MODELS ONCE (IMPORTANT FOR PERFORMANCE)
# =========================================================
try:
    sbp_model = joblib.load("sbp_model.pkl")
    dbp_model = joblib.load("dbp_model.pkl")
    scaler = joblib.load("bp_scaler.pkl")
    poly = joblib.load("bp_poly.pkl")
    BP_MODELS_LOADED = True
except:
    BP_MODELS_LOADED = False


# =========================================================
# BP Feature Extraction
# =========================================================
def extract_bp_features(signal_array, fps):

    hr = estimate_hr(signal_array, fps)
    pulse = get_pos_signal(signal_array)

    peaks, _ = find_peaks(pulse, distance=fps * 0.4)

    if len(peaks) < 2:
        return None

    rr_intervals = np.diff(peaks) / fps
    ptt = np.mean(rr_intervals)

    amplitude = np.std(pulse)

    green = signal_array[:, 1]
    ac = np.std(green)
    dc = np.mean(green)

    if dc < 1e-6:
        return None

    ac_dc_ratio = ac / dc

    return np.array([[hr, ptt, amplitude, ac_dc_ratio]])


# =========================================================
# 🎯 CORE PROCESSING FUNCTION (USED BY FASTAPI)
# =========================================================
def process_video_file(video_path, age=None, gender=None, height=None, weight=None):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Safety fallback
    if fps is None or fps <= 0:
        fps = 30

    face_validator = FaceValidator()

    signal = []
    frame_count = 0

    while cap.isOpened() and frame_count < TOTAL_FRAMES:

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        valid_face, landmarks = face_validator.validate(frame)

        if valid_face:
            roi = extract_roi(frame, landmarks)

            if roi.size != 0:
                rgb_val = extract_rgb_means(roi)

                if rgb_val is not None:
                    signal.append(rgb_val)

    cap.release()

    # =====================================================
    # SIGNAL CHECK
    # =====================================================
    if len(signal) < int(fps * 20):
        return None

    signal_array = np.array(signal)

    # ================== HR ==================
    final_hr = estimate_hr(signal_array, fps)

    # ================== RR ==================
    pos_signal = get_pos_signal(signal_array)
    final_rr = estimate_rr(pos_signal, fps)

    # ================== SpO2 ==================
    final_spo2 = estimate_spo2(signal_array, fps)

    # ================== HRV ==================
    hrv_metrics = estimate_hrv(pos_signal, fps)

    # ================== BP ==================
    sbp = None
    dbp = None

    if BP_MODELS_LOADED:
        try:
            features = extract_bp_features(signal_array, fps)

            if features is not None:
                features_scaled = scaler.transform(features)
                features_poly = poly.transform(features_scaled)

                sbp = sbp_model.predict(features_poly)[0]
                dbp = dbp_model.predict(features_poly)[0]

                # physiological clamp
                sbp = max(90, min(180, sbp))
                dbp = max(50, min(120, dbp))

        except:
            sbp = None
            dbp = None

    # =====================================================
    # CHART DATA: rPPG signal (downsampled for frontend)
    # =====================================================
    rppg_signal_data = []
    sample_step = max(1, len(pos_signal) // 500)  # cap at ~500 points
    for i in range(0, len(pos_signal), sample_step):
        rppg_signal_data.append({
            "time": round(float(i / fps), 3),
            "value": round(float(pos_signal[i]), 5)
        })

    # =====================================================
    # CHART DATA: HR pulse signal (normalized, for waveform)
    # =====================================================
    # Normalize POS signal for display
    norm_sig = pos_signal - np.mean(pos_signal)
    std_val = np.std(norm_sig)
    if std_val > 1e-8:
        norm_sig = norm_sig / std_val

    hr_signal_data = []
    hr_step = max(1, len(norm_sig) // 500)
    for i in range(0, len(norm_sig), hr_step):
        hr_signal_data.append({
            "time": round(float(i / fps), 3),
            "value": round(float(norm_sig[i]), 4)
        })

    # =====================================================
    # CHART DATA: windowed heart rate timeline
    # =====================================================
    hr_timeline = []
    window_size = int(fps * 10)
    step_size = int(fps * 5)

    for start in range(0, len(signal_array) - window_size, step_size):
        window = signal_array[start:start + window_size]
        try:
            bpm = estimate_hr(window, fps)
            if bpm and bpm > 0:
                time_sec = round(float((start + window_size // 2) / fps), 1)
                hr_timeline.append({
                    "time": time_sec,
                    "heartRate": round(float(bpm), 1)
                })
        except:
            pass

    # If no windowed estimates, provide at least the overall value
    if len(hr_timeline) == 0 and final_hr and final_hr > 0:
        total_duration = len(signal_array) / fps
        hr_timeline = [
            {"time": round(total_duration * 0.25, 1), "heartRate": round(float(final_hr), 1)},
            {"time": round(total_duration * 0.50, 1), "heartRate": round(float(final_hr), 1)},
            {"time": round(total_duration * 0.75, 1), "heartRate": round(float(final_hr), 1)},
        ]

    # =====================================================
    # RETURN CLEAN JSON
    # =====================================================
    return {
        "heart_rate_bpm": float(final_hr) if final_hr else None,
        "respiratory_rate_bpm": float(final_rr) if final_rr else None,
        "spo2_percent": float(final_spo2) if final_spo2 else None,
        "blood_pressure": {
            "systolic": float(sbp) if sbp else None,
            "diastolic": float(dbp) if dbp else None
        },
        "hrv": hrv_metrics,
        "chart_data": {
            "rppg_signal": rppg_signal_data,
            "hr_signal": hr_signal_data,
            "heart_rate_timeline": hr_timeline
        }
    }


# =========================================================
# OPTIONAL CLI MODE
# =========================================================
if __name__ == "__main__":
    import json

    path = input("Enter video path: ")

    result = process_video_file(path)

    if result is None:
        print("❌ Not enough signal.")
    else:
        print(json.dumps(result, indent=4))