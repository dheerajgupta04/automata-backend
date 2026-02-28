import cv2
import numpy as np
import json
import joblib
import os
import time
from scipy.signal import find_peaks

from config import TOTAL_FRAMES
from validation.face_validator import FaceValidator
from processing.roi_extractor import extract_roi
from processing.signal_processor import extract_rgb_means, calculate_sqi
from processing.hr_estimation import estimate_hr, get_pos_signal
from processing.rr_estimation import estimate_rr
from processing.hrv_estimation import estimate_hrv
from processing.spo2_estimation import estimate_spo2

# =========================================================
# LOAD BP MODELS
# =========================================================
try:
    sbp_model = joblib.load("sbp_model.pkl")
    dbp_model = joblib.load("dbp_model.pkl")
    scaler = joblib.load("bp_scaler.pkl")
    poly = joblib.load("bp_poly.pkl")
    BP_MODELS_LOADED = True
except:
    BP_MODELS_LOADED = False

def extract_bp_features(signal_array, fps):
    hr = estimate_hr(signal_array, fps)
    pulse = get_pos_signal(signal_array)
    peaks, _ = find_peaks(pulse, distance=int(fps * 0.4))
    if len(peaks) < 2: return None
    rr_intervals = np.diff(peaks) / fps
    ptt = np.mean(rr_intervals)
    amplitude = np.std(pulse)
    green = signal_array[:, 1]
    ac = np.std(green)
    dc = np.mean(green)
    if dc < 1e-6: return None
    return np.array([[hr, ptt, amplitude, ac / dc]])

def process_video_stream(video_path, age=None, gender=None, height=None, weight=None, update_callback=None):
    """
    Main processing pipeline for video files or webcam streams.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0: fps = 30.0

    face_validator = FaceValidator()
    signal = []
    frame_count = 0
    prev_bpm = None
    intermediate_results = []

    print(f"[*] Starting processing: {video_path} at {fps:.2f} FPS")

    while cap.isOpened() and frame_count < TOTAL_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        valid_face, landmarks = face_validator.validate(frame)
        if valid_face:
            rois = extract_roi(frame, landmarks)
            if rois:
                rgb_val = extract_rgb_means(rois)
                if rgb_val:
                    signal.append(rgb_val)

        # Intermediate updates every 100 frames starting from 500
        if frame_count >= 500 and frame_count % 100 == 0:
            if len(signal) > int(fps * 10):
                sig_arr = np.array(signal)
                current_hr = estimate_hr(sig_arr, fps, prev_bpm=prev_bpm)
                prev_bpm = current_hr
                
                pulse = get_pos_signal(sig_arr)
                current_rr = estimate_rr(pulse, fps)
                current_spo2 = estimate_spo2(sig_arr, fps)
                
                res = {
                    "frame": frame_count,
                    "heart_rate_bpm": round(current_hr, 1),
                    "respiratory_rate_bpm": round(current_rr, 1) if current_rr else None,
                    "spo2_percent": current_spo2
                }
                intermediate_results.append(res)
                if update_callback: update_callback(frame_count, res)
                print(f"[#] Frame {frame_count}: HR={current_hr:.1f}, RR={current_rr if current_rr else 'N/A'}")

    cap.release()

    if len(signal) < int(fps * 20):
        print("[!] Not enough signal collected.")
        return None

    signal_array = np.array(signal)
    pos_signal = get_pos_signal(signal_array)

    # FINAL ESTIMATIONS
    final_hr = estimate_hr(signal_array, fps, prev_bpm=prev_bpm)
    final_rr = estimate_rr(pos_signal, fps)
    final_spo2 = estimate_spo2(signal_array, fps)
    hrv_metrics = estimate_hrv(pos_signal, fps)

    # BP Estimation
    sbp, dbp = None, None
    if BP_MODELS_LOADED:
        try:
            feats = extract_bp_features(signal_array, fps)
            if feats is not None:
                poly_f = poly.transform(scaler.transform(feats))
                sbp = max(90, min(180, sbp_model.predict(poly_f)[0]))
                dbp = max(50, min(120, dbp_model.predict(poly_f)[0]))
        except: pass

    # Construct result in the format expected by API
    return {
        "heart_rate_bpm": float(round(final_hr, 2)),
        "respiratory_rate_bpm": float(round(final_rr, 2)) if final_rr else None,
        "spo2_percent": final_spo2,
        "blood_pressure": {
            "systolic": float(round(sbp, 1)) if sbp else None, 
            "diastolic": float(round(dbp, 1)) if dbp else None
        },
        "hrv": hrv_metrics
    }

# Compatibility wrapper for old callers
def process_video_file(video_path, age=None, gender=None, height=None, weight=None, update_callback=None):
    return process_video_stream(video_path, age, gender, height, weight, update_callback)

if __name__ == "__main__":
    print("\n=== PulseScanAI Enhanced rPPG Backend ===\n")
    path = input("Enter video path (or 0 for webcam): ")
    if path == "0": path = 0
    
    start_time = time.time()
    results = process_video_stream(path)
    
    if results:
        print("\n--- Final Results ---")
        print(json.dumps(results, indent=4))
        print(f"\nCompleted in {time.time() - start_time:.1f}s")
    else:
        print("❌ Processing failed.")