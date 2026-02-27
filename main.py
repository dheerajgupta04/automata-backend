# import cv2
# import numpy as np
# import json

# from config import *
# from utils.video_stream import get_stream
# from validation.brightness_validator import validate_brightness
# from validation.face_validator import FaceValidator
# from validation.movement_validator import MovementValidator
# from validation.fps_validator import validate_fps_and_duration
# from validation.quality_validator import validate_resolution, validate_camera_distance
# from processing.roi_extractor import extract_roi
# from processing.signal_processor import extract_rgb_means
# from processing.hr_estimation import estimate_hr, get_pos_signal
# from processing.rr_estimation import estimate_rr


# # -------------------------------------------------
# # Convert NumPy types to native Python (JSON safe)
# # -------------------------------------------------
# def convert_to_native(obj):
#     import numpy as np

#     if isinstance(obj, np.generic):
#         return obj.item()

#     if isinstance(obj, dict):
#         return {k: convert_to_native(v) for k, v in obj.items()}

#     if isinstance(obj, list):
#         return [convert_to_native(i) for i in obj]

#     return obj


# def main():

#     print("\n===== PulseScanAI (Automata) rPPG System =====\n")

#     choice = input("Enter 1 to Record (Webcam) or 2 to Upload Video: ")

#     if choice == "1":
#         cap = get_stream(0)
#     elif choice == "2":
#         path = input("Enter video path: ")
#         cap = get_stream(path)
#     else:
#         print("Invalid choice.")
#         return

#     fps_validation = validate_fps_and_duration(cap)

#     if not fps_validation["valid"]:
#         print("\n❌ FPS / Duration validation failed")
#         print(fps_validation)
#         cap.release()
#         return

#     fps = fps_validation["fps"]

#     face_validator = FaceValidator()
#     movement_validator = MovementValidator()

#     validation_report = {
#         "fps_duration": fps_validation,
#         "brightness": [],
#         "resolution": [],
#         "movement": [],
#         "face": [],
#         "distance": []
#     }

#     signal = []
#     hr_estimations = []
#     frame_count = 0

#     print("\nProcessing stream...\n")

#     while cap.isOpened() and frame_count < TOTAL_FRAMES:

#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1

#         # -------------------------
#         # Validation (1–500)
#         # -------------------------
#         if frame_count <= VALIDATION_FRAMES and frame_count % 2 == 0:

#             validation_report["resolution"].append(validate_resolution(frame))
#             validation_report["brightness"].append(validate_brightness(frame))

#             valid_face, landmarks = face_validator.validate(frame)
#             validation_report["face"].append(valid_face)

#             if valid_face:
#                 validation_report["movement"].append(
#                     movement_validator.validate(landmarks)
#                 )

#                 validation_report["distance"].append(
#                     validate_camera_distance(frame, landmarks)
#                 )

#         # -------------------------
#         # Signal Collection (1–900)
#         # -------------------------
#         valid_face, landmarks = face_validator.validate(frame)

#         if valid_face:
#             roi = extract_roi(frame, landmarks)
#             if roi.size != 0:
#                 rgb_val = extract_rgb_means(roi)
#                 if rgb_val is not None:
#                     signal.append(rgb_val)

#         # -------------------------
#         # Intermediate HR
#         # -------------------------
#         if frame_count in [600, 700, 800, 900]:

#             if len(signal) > int(fps * 8):

#                 try:
#                     bpm = estimate_hr(np.array(signal), fps)

#                     hr_estimations.append({
#                         "frame": frame_count,
#                         "bpm": float(bpm)
#                     })

#                     print(f"HR at frame {frame_count}: {bpm:.2f} BPM")

#                 except:
#                     pass

#     cap.release()

#     # -------------------------
#     # Final HR + RR
#     # -------------------------
#     if len(signal) < int(fps * 10):
#         print("\n❌ Not enough signal for final estimation")
#         print("Collected samples:", len(signal))
#         return

#     signal_array = np.array(signal)

#     # -------- HR (UNCHANGED) --------
#     final_bpm = estimate_hr(signal_array, fps)

#     # -------- RR (STABILIZED) --------
#     # Use only last 25 seconds of signal for RR
#     rr_window = int(fps * 25)

#     if len(signal_array) > rr_window:
#         rr_signal_part = signal_array[-rr_window:]
#     else:
#         rr_signal_part = signal_array

#     pos_pulse = get_pos_signal(rr_signal_part)
#     final_rr = estimate_rr(pos_pulse, fps)

#     final_output = {
#         "validation": validation_report,
#         "hr_estimations": hr_estimations,
#         "final_results": {
#             "heart_rate_bpm": float(final_bpm),
#             "respiratory_rate_bpm": float(final_rr) if final_rr is not None else None
#         }
#     }

#     print("\nFinal Result:\n")

#     clean_output = convert_to_native(final_output)
#     print(json.dumps(clean_output, indent=4))

#     print("\n✅ Processing Complete\n")


# if __name__ == "__main__":
#     main()




# import cv2
# import numpy as np
# import json

# from config import *
# from utils.video_stream import get_stream
# from validation.brightness_validator import validate_brightness
# from validation.face_validator import FaceValidator
# from validation.movement_validator import MovementValidator
# from validation.fps_validator import validate_fps_and_duration
# from validation.quality_validator import validate_resolution, validate_camera_distance
# from processing.roi_extractor import extract_roi
# from processing.signal_processor import extract_rgb_means
# from processing.hr_estimation import estimate_hr, get_pos_signal
# from processing.rr_estimation import estimate_rr
# from processing.hrv_estimation import estimate_hrv   # ✅ NEW


# # -------------------------------------------------
# # Convert NumPy types to native Python (JSON safe)
# # -------------------------------------------------
# def convert_to_native(obj):
#     import numpy as np

#     if isinstance(obj, np.generic):
#         return obj.item()

#     if isinstance(obj, dict):
#         return {k: convert_to_native(v) for k, v in obj.items()}

#     if isinstance(obj, list):
#         return [convert_to_native(i) for i in obj]

#     return obj


# def main():

#     print("\n===== PulseScanAI (Automata) rPPG System =====\n")

#     choice = input("Enter 1 to Record (Webcam) or 2 to Upload Video: ")

#     if choice == "1":
#         cap = get_stream(0)
#     elif choice == "2":
#         path = input("Enter video path: ")
#         cap = get_stream(path)
#     else:
#         print("Invalid choice.")
#         return

#     fps_validation = validate_fps_and_duration(cap)

#     if not fps_validation["valid"]:
#         print("\n❌ FPS / Duration validation failed")
#         print(fps_validation)
#         cap.release()
#         return

#     fps = fps_validation["fps"]

#     face_validator = FaceValidator()
#     movement_validator = MovementValidator()

#     validation_report = {
#         "fps_duration": fps_validation,
#         "brightness": [],
#         "resolution": [],
#         "movement": [],
#         "face": [],
#         "distance": []
#     }

#     signal = []
#     hr_estimations = []
#     frame_count = 0

#     print("\nProcessing stream...\n")

#     while cap.isOpened() and frame_count < TOTAL_FRAMES:

#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1

#         # -------------------------
#         # Validation (1–500)
#         # -------------------------
#         if frame_count <= VALIDATION_FRAMES and frame_count % 2 == 0:

#             validation_report["resolution"].append(validate_resolution(frame))
#             validation_report["brightness"].append(validate_brightness(frame))

#             valid_face, landmarks = face_validator.validate(frame)
#             validation_report["face"].append(valid_face)

#             if valid_face:
#                 validation_report["movement"].append(
#                     movement_validator.validate(landmarks)
#                 )

#                 validation_report["distance"].append(
#                     validate_camera_distance(frame, landmarks)
#                 )

#         # -------------------------
#         # Signal Collection (1–900)
#         # -------------------------
#         valid_face, landmarks = face_validator.validate(frame)

#         if valid_face:
#             roi = extract_roi(frame, landmarks)
#             if roi.size != 0:
#                 rgb_val = extract_rgb_means(roi)
#                 if rgb_val is not None:
#                     signal.append(rgb_val)

#         # -------------------------
#         # Intermediate HR
#         # -------------------------
#         if frame_count in [600, 700, 800, 900]:

#             if len(signal) > int(fps * 8):

#                 try:
#                     bpm = estimate_hr(np.array(signal), fps)

#                     hr_estimations.append({
#                         "frame": frame_count,
#                         "bpm": float(bpm)
#                     })

#                     print(f"HR at frame {frame_count}: {bpm:.2f} BPM")

#                 except:
#                     pass

#     cap.release()

#     # -------------------------
#     # Final Estimation
#     # -------------------------
#     if len(signal) < int(fps * 20):
#         print("\n❌ Not enough signal for final estimation")
#         print("Collected samples:", len(signal))
#         return

#     signal_array = np.array(signal)

#     # -------- HR --------
#     final_bpm = estimate_hr(signal_array, fps)

#     # -------- RR (last 25 sec stabilization) --------
#     rr_window = int(fps * 25)

#     if len(signal_array) > rr_window:
#         rr_signal_part = signal_array[-rr_window:]
#     else:
#         rr_signal_part = signal_array

#     pos_pulse_rr = get_pos_signal(rr_signal_part)
#     final_rr = estimate_rr(pos_pulse_rr, fps)

#     # -------- HRV (full POS pulse) --------
#     pos_pulse_full = get_pos_signal(signal_array)
#     hrv_metrics = estimate_hrv(pos_pulse_full, fps)

#     final_output = {
#         "validation": validation_report,
#         "hr_estimations": hr_estimations,
#         "final_results": {
#             "heart_rate_bpm": float(final_bpm),
#             "respiratory_rate_bpm": float(final_rr) if final_rr is not None else None,
#             "hrv": hrv_metrics
#         }
#     }

#     print("\nFinal Result:\n")

#     clean_output = convert_to_native(final_output)
#     print(json.dumps(clean_output, indent=4))

#     print("\n✅ Processing Complete\n")


# if __name__ == "__main__":
#     main()
# import cv2
# import numpy as np
# import json

# from config import *
# from utils.video_stream import get_stream
# from validation.brightness_validator import validate_brightness
# from validation.face_validator import FaceValidator
# from validation.movement_validator import MovementValidator
# from validation.fps_validator import validate_fps_and_duration
# from validation.quality_validator import validate_resolution, validate_camera_distance
# from processing.roi_extractor import extract_roi
# from processing.signal_processor import extract_rgb_means
# from processing.hr_estimation import estimate_hr, get_pos_signal
# from processing.rr_estimation import estimate_rr
# from processing.hrv_estimation import estimate_hrv
# from processing.spo2_estimation import estimate_spo2   # ✅ NEW


# # -------------------------------------------------
# # Convert NumPy types to native Python (JSON safe)
# # -------------------------------------------------
# def convert_to_native(obj):
#     import numpy as np

#     if isinstance(obj, np.generic):
#         return obj.item()

#     if isinstance(obj, dict):
#         return {k: convert_to_native(v) for k, v in obj.items()}

#     if isinstance(obj, list):
#         return [convert_to_native(i) for i in obj]

#     return obj


# def main():

#     print("\n===== PulseScanAI (Automata) rPPG System =====\n")

#     choice = input("Enter 1 to Record (Webcam) or 2 to Upload Video: ")

#     if choice == "1":
#         cap = get_stream(0)
#     elif choice == "2":
#         path = input("Enter video path: ")
#         cap = get_stream(path)
#     else:
#         print("Invalid choice.")
#         return

#     fps_validation = validate_fps_and_duration(cap)

#     if not fps_validation["valid"]:
#         print("\n❌ FPS / Duration validation failed")
#         print(fps_validation)
#         cap.release()
#         return

#     fps = fps_validation["fps"]

#     face_validator = FaceValidator()
#     movement_validator = MovementValidator()

#     validation_report = {
#         "fps_duration": fps_validation,
#         "brightness": [],
#         "resolution": [],
#         "movement": [],
#         "face": [],
#         "distance": []
#     }

#     signal = []
#     hr_estimations = []
#     frame_count = 0

#     print("\nProcessing stream...\n")

#     while cap.isOpened() and frame_count < TOTAL_FRAMES:

#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1

#         # -------------------------
#         # Validation (1–500)
#         # -------------------------
#         if frame_count <= VALIDATION_FRAMES and frame_count % 2 == 0:

#             validation_report["resolution"].append(validate_resolution(frame))
#             validation_report["brightness"].append(validate_brightness(frame))

#             valid_face, landmarks = face_validator.validate(frame)
#             validation_report["face"].append(valid_face)

#             if valid_face:
#                 validation_report["movement"].append(
#                     movement_validator.validate(landmarks)
#                 )

#                 validation_report["distance"].append(
#                     validate_camera_distance(frame, landmarks)
#                 )

#         # -------------------------
#         # Signal Collection (1–900)
#         # -------------------------
#         valid_face, landmarks = face_validator.validate(frame)

#         if valid_face:
#             roi = extract_roi(frame, landmarks)
#             if roi.size != 0:
#                 rgb_val = extract_rgb_means(roi)
#                 if rgb_val is not None:
#                     signal.append(rgb_val)

#         # -------------------------
#         # Intermediate HR
#         # -------------------------
#         if frame_count in [600, 700, 800, 900]:

#             if len(signal) > int(fps * 8):

#                 try:
#                     bpm = estimate_hr(np.array(signal), fps)

#                     hr_estimations.append({
#                         "frame": frame_count,
#                         "bpm": float(bpm)
#                     })

#                     print(f"HR at frame {frame_count}: {bpm:.2f} BPM")

#                 except:
#                     pass

#     cap.release()

#     # -------------------------
#     # Final Estimation
#     # -------------------------
#     if len(signal) < int(fps * 20):
#         print("\n❌ Not enough signal for final estimation")
#         print("Collected samples:", len(signal))
#         return

#     signal_array = np.array(signal)

#     # -------- HR --------
#     final_bpm = estimate_hr(signal_array, fps)

#     # -------- RR (last 25 sec stabilization) --------
#     rr_window = int(fps * 25)

#     if len(signal_array) > rr_window:
#         rr_signal_part = signal_array[-rr_window:]
#     else:
#         rr_signal_part = signal_array

#     pos_pulse_rr = get_pos_signal(rr_signal_part)
#     final_rr = estimate_rr(pos_pulse_rr, fps)

#     # -------- HRV --------
#     pos_pulse_full = get_pos_signal(signal_array)
#     hrv_metrics = estimate_hrv(pos_pulse_full, fps)

#     # -------- SpO₂ --------
#     final_spo2 = estimate_spo2(signal_array, fps)

#     final_output = {
#         "validation": validation_report,
#         "hr_estimations": hr_estimations,
#         "final_results": {
#             "heart_rate_bpm": float(final_bpm),
#             "respiratory_rate_bpm": float(final_rr) if final_rr is not None else None,
#             "spo2_percent": float(final_spo2) if final_spo2 is not None else None,
#             "hrv": hrv_metrics
#         }
#     }

#     print("\nFinal Result:\n")

#     clean_output = convert_to_native(final_output)
#     print(json.dumps(clean_output, indent=4))

#     print("\n✅ Processing Complete\n")


# if __name__ == "__main__":
#     main()
# import cv2
# import numpy as np
# import json
# import joblib
# from scipy.signal import find_peaks

# from config import *
# from utils.video_stream import get_stream
# from validation.brightness_validator import validate_brightness
# from validation.face_validator import FaceValidator
# from validation.movement_validator import MovementValidator
# from validation.fps_validator import validate_fps_and_duration
# from validation.quality_validator import validate_resolution, validate_camera_distance
# from processing.roi_extractor import extract_roi
# from processing.signal_processor import extract_rgb_means
# from processing.hr_estimation import estimate_hr, get_pos_signal
# from processing.rr_estimation import estimate_rr
# from processing.hrv_estimation import estimate_hrv
# from processing.spo2_estimation import estimate_spo2


# # -------------------------------------------------
# # Convert NumPy types to native Python (JSON safe)
# # -------------------------------------------------
# def convert_to_native(obj):
#     if isinstance(obj, np.generic):
#         return obj.item()
#     if isinstance(obj, dict):
#         return {k: convert_to_native(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [convert_to_native(i) for i in obj]
#     return obj


# # -------------------------------------------------
# # BP Feature Extraction (same as training)
# # -------------------------------------------------
# def extract_bp_features(signal_array, fps):

#     hr = estimate_hr(signal_array, fps)

#     pulse = get_pos_signal(signal_array)

#     peaks, _ = find_peaks(pulse, distance=fps * 0.4)

#     if len(peaks) < 2:
#         return None

#     rr_intervals = np.diff(peaks) / fps
#     ptt = np.mean(rr_intervals)

#     amplitude = np.std(pulse)

#     green = signal_array[:, 1]
#     ac = np.std(green)
#     dc = np.mean(green)

#     if dc < 1e-6:
#         return None

#     ac_dc_ratio = ac / dc

#     return np.array([[hr, ptt, amplitude, ac_dc_ratio]])


# def main():

#     print("\n===== PulseScanAI (Automata) rPPG System =====\n")

#     choice = input("Enter 1 to Record (Webcam) or 2 to Upload Video: ")

#     if choice == "1":
#         cap = get_stream(0)
#     elif choice == "2":
#         path = input("Enter video path: ")
#         cap = get_stream(path)
#     else:
#         print("Invalid choice.")
#         return

#     fps_validation = validate_fps_and_duration(cap)

#     if not fps_validation["valid"]:
#         print("\n❌ FPS / Duration validation failed")
#         print(fps_validation)
#         cap.release()
#         return

#     fps = fps_validation["fps"]

#     face_validator = FaceValidator()
#     movement_validator = MovementValidator()

#     validation_report = {
#         "fps_duration": fps_validation,
#         "brightness": [],
#         "resolution": [],
#         "movement": [],
#         "face": [],
#         "distance": []
#     }

#     signal = []
#     frame_count = 0

#     print("\nProcessing stream...\n")

#     while cap.isOpened() and frame_count < TOTAL_FRAMES:

#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1

#         # ---------- Validation ----------
#         if frame_count <= VALIDATION_FRAMES and frame_count % 2 == 0:

#             validation_report["resolution"].append(validate_resolution(frame))
#             validation_report["brightness"].append(validate_brightness(frame))

#             valid_face, landmarks = face_validator.validate(frame)
#             validation_report["face"].append(valid_face)

#             if valid_face:
#                 validation_report["movement"].append(
#                     movement_validator.validate(landmarks)
#                 )
#                 validation_report["distance"].append(
#                     validate_camera_distance(frame, landmarks)
#                 )

#         # ---------- Signal Collection ----------
#         valid_face, landmarks = face_validator.validate(frame)

#         if valid_face:
#             roi = extract_roi(frame, landmarks)
#             if roi.size != 0:
#                 rgb_val = extract_rgb_means(roi)
#                 if rgb_val is not None:
#                     signal.append(rgb_val)

#     cap.release()

#     if len(signal) < int(fps * 20):
#         print("\n❌ Not enough signal for final estimation")
#         return

#     signal_array = np.array(signal)

#     # ================== HR ==================
#     final_bpm = estimate_hr(signal_array, fps)

#     # ================== RR ==================
#     rr_window = int(fps * 25)
#     rr_signal_part = signal_array[-rr_window:] if len(signal_array) > rr_window else signal_array
#     pos_rr = get_pos_signal(rr_signal_part)
#     final_rr = estimate_rr(pos_rr, fps)

#     # ================== HRV ==================
#     pos_full = get_pos_signal(signal_array)
#     hrv_metrics = estimate_hrv(pos_full, fps)

#     # ================== SpO2 ==================
#     final_spo2 = estimate_spo2(signal_array, fps)

#     # ================== BP ==================
#     try:
#         sbp_model = joblib.load("sbp_model.pkl")
#         dbp_model = joblib.load("dbp_model.pkl")
#         scaler = joblib.load("bp_scaler.pkl")
#         poly = joblib.load("bp_poly.pkl")

#         features = extract_bp_features(signal_array, fps)

#         if features is not None:

#             features_scaled = scaler.transform(features)
#             features_poly = poly.transform(features_scaled)

#             sbp = sbp_model.predict(features_poly)[0]
#             dbp = dbp_model.predict(features_poly)[0]

#             # physiological clamp
#             sbp = max(90, min(180, sbp))
#             dbp = max(50, min(120, dbp))

#         else:
#             sbp = None
#             dbp = None

#     except Exception as e:
#         print("BP Error:", e)
#         sbp = None
#         dbp = None

#     # ================== FINAL OUTPUT ==================
#     final_output = {
#         "validation": validation_report,
#         "final_results": {
#             "heart_rate_bpm": float(final_bpm),
#             "respiratory_rate_bpm": float(final_rr) if final_rr else None,
#             "spo2_percent": float(final_spo2) if final_spo2 else None,
#             "blood_pressure": {
#                 "systolic": float(sbp) if sbp else None,
#                 "diastolic": float(dbp) if dbp else None
#             },
#             "hrv": hrv_metrics
#         }
#     }

#     print("\nFinal Result:\n")
#     print(json.dumps(convert_to_native(final_output), indent=4))
#     print("\n✅ Processing Complete\n")


# if __name__ == "__main__":
#     main()
# import cv2
# import numpy as np
# import json
# import joblib
# from scipy.signal import find_peaks

# from config import *
# from utils.video_stream import get_stream
# from validation.face_validator import FaceValidator
# from processing.roi_extractor import extract_roi
# from processing.signal_processor import extract_rgb_means
# from processing.hr_estimation import estimate_hr, get_pos_signal
# from processing.rr_estimation import estimate_rr
# from processing.hrv_estimation import estimate_hrv
# from processing.spo2_estimation import estimate_spo2


# # -------------------------------------------------
# # Convert NumPy types to native Python (JSON safe)
# # -------------------------------------------------
# def convert_to_native(obj):
#     if isinstance(obj, np.generic):
#         return obj.item()
#     if isinstance(obj, dict):
#         return {k: convert_to_native(v) for k, v in obj.items()}
#     if isinstance(obj, list):
#         return [convert_to_native(i) for i in obj]
#     return obj


# # -------------------------------------------------
# # BP Feature Extraction
# # -------------------------------------------------
# def extract_bp_features(signal_array, fps):

#     hr = estimate_hr(signal_array, fps)

#     pulse = get_pos_signal(signal_array)
#     peaks, _ = find_peaks(pulse, distance=fps * 0.4)

#     if len(peaks) < 2:
#         return None

#     rr_intervals = np.diff(peaks) / fps
#     ptt = np.mean(rr_intervals)

#     amplitude = np.std(pulse)

#     green = signal_array[:, 1]
#     ac = np.std(green)
#     dc = np.mean(green)

#     if dc < 1e-6:
#         return None

#     ac_dc_ratio = ac / dc

#     return np.array([[hr, ptt, amplitude, ac_dc_ratio]])


# # -------------------------------------------------
# # Core Processing Function (USED BY FASTAPI)
# # -------------------------------------------------
# def process_video_file(video_path, update_callback=None):

#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     face_validator = FaceValidator()
#     signal = []
#     frame_count = 0

#     while cap.isOpened() and frame_count < TOTAL_FRAMES:

#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1

#         valid_face, landmarks = face_validator.validate(frame)

#         if valid_face:
#             roi = extract_roi(frame, landmarks)
#             if roi.size != 0:
#                 rgb_val = extract_rgb_means(roi)
#                 if rgb_val is not None:
#                     signal.append(rgb_val)

#         # 🔥 Live Updates from 500–900
#         if update_callback and frame_count >= 500 and frame_count % 100 == 0:

#             if len(signal) > int(fps * 8):

#                 signal_array = np.array(signal)

#                 hr = estimate_hr(signal_array, fps)

#                 rr_window = int(fps * 25)
#                 rr_signal = signal_array[-rr_window:] if len(signal_array) > rr_window else signal_array
#                 pos_rr = get_pos_signal(rr_signal)
#                 rr = estimate_rr(pos_rr, fps)

#                 spo2 = estimate_spo2(signal_array, fps)

#                 update_callback(frame_count, {
#                     "heart_rate_bpm": float(hr),
#                     "respiratory_rate_bpm": float(rr) if rr else None,
#                     "spo2_percent": float(spo2) if spo2 else None
#                 })

#     cap.release()

#     if len(signal) < int(fps * 20):
#         return None

#     signal_array = np.array(signal)

#     # ===== FINAL HR =====
#     final_bpm = estimate_hr(signal_array, fps)

#     # ===== FINAL RR =====
#     pos_full = get_pos_signal(signal_array)
#     final_rr = estimate_rr(pos_full, fps)

#     # ===== FINAL SpO2 =====
#     final_spo2 = estimate_spo2(signal_array, fps)

#     # ===== FINAL HRV =====
#     hrv_metrics = estimate_hrv(pos_full, fps)

#     # ===== FINAL BP =====
#     try:
#         sbp_model = joblib.load("sbp_model.pkl")
#         dbp_model = joblib.load("dbp_model.pkl")
#         scaler = joblib.load("bp_scaler.pkl")
#         poly = joblib.load("bp_poly.pkl")

#         features = extract_bp_features(signal_array, fps)

#         if features is not None:
#             features_scaled = scaler.transform(features)
#             features_poly = poly.transform(features_scaled)

#             sbp = sbp_model.predict(features_poly)[0]
#             dbp = dbp_model.predict(features_poly)[0]

#             sbp = max(90, min(180, sbp))
#             dbp = max(50, min(120, dbp))
#         else:
#             sbp = None
#             dbp = None

#     except:
#         sbp = None
#         dbp = None

#     return {
#         "heart_rate_bpm": float(final_bpm),
#         "respiratory_rate_bpm": float(final_rr) if final_rr else None,
#         "spo2_percent": float(final_spo2) if final_spo2 else None,
#         "blood_pressure": {
#             "systolic": float(sbp) if sbp else None,
#             "diastolic": float(dbp) if dbp else None
#         },
#         "hrv": hrv_metrics
#     }


# # -------------------------------------------------
# # Standalone CLI Mode (Optional)
# # -------------------------------------------------
# def main():

#     print("\n===== PulseScanAI CLI Mode =====\n")

#     path = input("Enter video path: ")

#     result = process_video_file(path)

#     if result is None:
#         print("❌ Not enough signal.")
#         return

#     print("\nFinal Result:\n")
#     print(json.dumps(convert_to_native(result), indent=4))


# if __name__ == "__main__":
#     main()
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
        "hrv": hrv_metrics
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