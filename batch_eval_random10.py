import os
import cv2
import random
import numpy as np
import pandas as pd
from tabulate import tabulate

from config import TOTAL_FRAMES
from validation.face_validator import FaceValidator
from processing.roi_extractor import extract_roi
from processing.signal_processor import extract_rgb_means
from processing.hr_estimation import estimate_hr, get_pos_signal
from processing.rr_estimation import estimate_rr


DATASET_PATH = r"C:\Users\shash\PulseScanAI\Dataset"
EXCEL_PATH = r"C:\Users\shash\PulseScanAI\Dataset\Ground_Truth.xlsx"


def estimate_video_parameters(video_path):

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

    if len(signal) < int(fps * 8):
        return None, None

    signal = np.array(signal)

    try:
        # HR (UNCHANGED)
        hr = estimate_hr(signal, fps)

        # RR from POS pulse (NEW)
        pos_pulse = get_pos_signal(signal)
        rr = estimate_rr(pos_pulse, fps)

    except:
        return None, None

    return round(float(hr), 2), round(float(rr), 2) if rr else None


def main():

    df = pd.read_excel(EXCEL_PATH)

    all_videos = list(range(1, 101))
    random_videos = random.sample(all_videos, 10)

    results = []
    total_error = 0
    valid_cases = 0

    print("\n===== Random 10 Video Evaluation =====\n")
    print("Selected Videos:", random_videos)
    print()

    for i in random_videos:

        video_file = os.path.join(DATASET_PATH, f"{i}.mp4")

        if not os.path.exists(video_file):
            print(f"Video {i}.mp4 not found")
            continue

        print(f"Processing Video {i}...")

        estimated_hr, estimated_rr = estimate_video_parameters(video_file)

        actual_hr = df.loc[df["pyth"] == i, "Heart_Rate"].values

        if len(actual_hr) == 0:
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
            error,
            estimated_rr
        ])

    print("\n\n===== Evaluation Results =====\n")

    print(tabulate(
        results,
        headers=["Video_No", "Estimated_HR", "Actual_HR", "Abs_Error", "Estimated_RR"],
        tablefmt="grid"
    ))

    if valid_cases > 0:
        mae = round(total_error / valid_cases, 2)
        print(f"\nMean Absolute Error (MAE): {mae} BPM")
    else:
        print("\nNo valid HR cases processed.")


if __name__ == "__main__":
    main()