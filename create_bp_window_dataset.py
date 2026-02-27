import os
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from config import TOTAL_FRAMES
from validation.face_validator import FaceValidator
from processing.roi_extractor import extract_roi
from processing.signal_processor import extract_rgb_means
from processing.hr_estimation import estimate_hr, get_pos_signal


DATASET_PATH = r"C:\Users\shash\PulseScanAI\Dataset"
EXCEL_PATH = r"C:\Users\shash\PulseScanAI\Dataset\Ground_Truth.xlsx"

WINDOW_SECONDS = 10
STEP_SECONDS = 5


def extract_features_from_window(rgb_window, fps):

    signal = np.array(rgb_window)

    # ---- Heart Rate ----
    hr = estimate_hr(signal, fps)

    # ---- POS pulse signal ----
    pulse = get_pos_signal(signal)

    # ---- Peak detection ----
    peaks, _ = find_peaks(pulse, distance=fps * 0.4)

    if len(peaks) < 2:
        return None

    rr_intervals = np.diff(peaks) / fps
    ptt = np.mean(rr_intervals)

    amplitude = np.std(pulse)

    green = signal[:, 1]

    ac = np.std(green)
    dc = np.mean(green)

    if dc < 1e-6:
        return None

    ac_dc_ratio = ac / dc

    return {
        "heart_rate": float(hr),
        "ptt": float(ptt),
        "amplitude": float(amplitude),
        "ac_dc_ratio": float(ac_dc_ratio)
    }


def main():

    df = pd.read_excel(EXCEL_PATH)

    face_validator = FaceValidator()
    all_rows = []

    for i in range(1, 101):

        video_path = os.path.join(DATASET_PATH, f"{i}.mp4")

        if not os.path.exists(video_path):
            continue

        print(f"Processing Video {i}...")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
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
                        frames.append(rgb_val)

        cap.release()

        if len(frames) < fps * WINDOW_SECONDS:
            continue

        # ✅ Correct Column Names
        systolic = df.loc[df["pyth"] == i, "Systolic"].values[0]
        diastolic = df.loc[df["pyth"] == i, "Diastolic"].values[0]

        window_size = int(fps * WINDOW_SECONDS)
        step_size = int(fps * STEP_SECONDS)

        for start in range(0, len(frames) - window_size, step_size):

            window = frames[start:start + window_size]

            features = extract_features_from_window(window, fps)

            if features is None:
                continue

            features["systolic"] = float(systolic)
            features["diastolic"] = float(diastolic)

            all_rows.append(features)

    dataset = pd.DataFrame(all_rows)
    dataset.to_csv("bp_windowed_dataset.csv", index=False)

    print("\nDataset saved as bp_windowed_dataset.csv")
    print("Total samples:", len(dataset))


if __name__ == "__main__":
    main()