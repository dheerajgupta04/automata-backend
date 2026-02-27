import os
import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate

from config import TOTAL_FRAMES
from validation.face_validator import FaceValidator
from processing.roi_extractor import extract_roi
from processing.signal_processor import extract_rgb_means
from processing.hr_estimation import estimate_hr


DATASET_PATH = r"C:\Users\shash\PulseScanAI\Dataset"
EXCEL_PATH = r"C:\Users\shash\PulseScanAI\Dataset\Ground_Truth.xlsx"  # <-- change if needed


def estimate_video_hr(video_path):
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
                signal.append(rgb_val)

    cap.release()

    if len(signal) < int(fps * 8):
        return None  # insufficient signal

    signal = np.array(signal)
    bpm = estimate_hr(signal, fps)

    return round(bpm, 2)


def main():

    df = pd.read_excel(EXCEL_PATH)

    results = []

    total_error = 0
    valid_cases = 0

    for i in range(1, 101):

        video_file = os.path.join(DATASET_PATH, f"{i}.mp4")

        if not os.path.exists(video_file):
            print(f"Video {i}.mp4 not found")
            continue

        print(f"Processing Video {i}...")

        estimated_hr = estimate_video_hr(video_file)

        actual_hr = df.loc[df["pyth"] == i, "Heart_Rate"].values

        if len(actual_hr) == 0:
            print(f"No actual HR found for video {i}")
            continue

        actual_hr = float(actual_hr[0])

        if estimated_hr is None:
            error = None
        else:
            error = round(abs(estimated_hr - actual_hr), 2)
            total_error += error
            valid_cases += 1

        results.append([
            i,
            estimated_hr,
            actual_hr,
            error
        ])

    print("\n\n===== Evaluation Results =====\n")
    print(tabulate(
        results,
        headers=["Video_No", "Estimated_HR", "Actual_HR", "Absolute_Error"],
        tablefmt="grid"
    ))

    if valid_cases > 0:
        mae = round(total_error / valid_cases, 2)
        print(f"\nMean Absolute Error (MAE): {mae} BPM")
    else:
        print("\nNo valid cases processed.")


if __name__ == "__main__":
    main()