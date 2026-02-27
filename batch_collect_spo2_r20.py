import os
import cv2
import random
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.signal import butter, filtfilt

from config import TOTAL_FRAMES
from validation.face_validator import FaceValidator
from processing.roi_extractor import extract_roi
from processing.signal_processor import extract_rgb_means


DATASET_PATH = r"C:\Users\shash\PulseScanAI\Dataset"
EXCEL_PATH = r"C:\Users\shash\PulseScanAI\Dataset\Ground_Truth.xlsx"


# ---------------------------
# Bandpass for AC extraction
# ---------------------------
def bandpass(signal, fps, low=0.7, high=3.0, order=3):
    nyquist = 0.5 * fps
    low /= nyquist
    high /= nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


# ---------------------------
# Extract R value
# ---------------------------
def extract_r_value(video_path):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

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

    if len(signal) < fps * 20:
        return None, None

    signal = np.array(signal)

    red = signal[:, 2]
    green = signal[:, 1]

    red_dc = np.mean(red)
    green_dc = np.mean(green)

    red_ac = bandpass(red, fps)
    green_ac = bandpass(green, fps)

    red_ac_amp = np.std(red_ac)
    green_ac_amp = np.std(green_ac)

    if red_dc < 1e-6 or green_dc < 1e-6 or green_ac_amp < 1e-6:
        return None, None

    R = (red_ac_amp / red_dc) / (green_ac_amp / green_dc)

    # Current predicted using old formula
    predicted_spo2 = 110 - 25 * R

    return float(R), float(predicted_spo2)


# ---------------------------
# Main
# ---------------------------
def main():

    df = pd.read_excel(EXCEL_PATH)

    all_videos = list(range(1, 101))
    random_videos = random.sample(all_videos, 20)

    results = []

    print("\n===== SpO2 Calibration Data (Random 20 Videos) =====\n")
    print("Selected Videos:", random_videos)
    print()

    for i in random_videos:

        video_file = os.path.join(DATASET_PATH, f"{i}.mp4")

        if not os.path.exists(video_file):
            continue

        print(f"Processing Video {i}...")

        R, predicted = extract_r_value(video_file)

        gt_row = df.loc[df["pyth"] == i, "spO2"].values

        if len(gt_row) == 0:
            continue

        gt_spo2 = float(gt_row[0])

        if R is None:
            results.append([i, None, gt_spo2, None])
        else:
            results.append([
                i,
                round(R, 4),
                gt_spo2,
                round(predicted, 2)
            ])

    print("\n\n===== Calibration Dataset =====\n")

    print(tabulate(
        results,
        headers=["Video_No", "R_Value", "GT_spO2", "Current_Predicted"],
        tablefmt="grid"
    ))


if __name__ == "__main__":
    main()