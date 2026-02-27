import os
import cv2
import random
import numpy as np
from tabulate import tabulate

from config import TOTAL_FRAMES
from validation.face_validator import FaceValidator
from processing.roi_extractor import extract_roi
from processing.signal_processor import extract_rgb_means
from processing.hr_estimation import get_pos_signal
from processing.hrv_estimation import estimate_hrv


DATASET_PATH = r"C:\Users\shash\PulseScanAI\Dataset"


def extract_pos_pulse(video_path):

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

    if len(signal) < int(fps * 20):
        return None, None

    signal_array = np.array(signal)
    pos_pulse = get_pos_signal(signal_array)

    return pos_pulse, fps


def main():

    all_videos = list(range(1, 101))
    random_videos = random.sample(all_videos, 15)

    results = []

    print("\n===== HRV Evaluation (Random 15 Videos) =====\n")
    print("Selected Videos:", random_videos)
    print()

    for i in random_videos:

        video_file = os.path.join(DATASET_PATH, f"{i}.mp4")

        if not os.path.exists(video_file):
            print(f"Video {i}.mp4 not found")
            continue

        print(f"Processing Video {i}...")

        pos_pulse, fps = extract_pos_pulse(video_file)

        if pos_pulse is None:
            results.append([i, None, None, None, None, None])
            continue

        hrv = estimate_hrv(pos_pulse, fps)

        if hrv is None:
            results.append([i, None, None, None, None, None])
            continue

        results.append([
            i,
            round(hrv["sdnn_ms"], 2),
            round(hrv["rmssd_ms"], 2),
            round(hrv["pnn50_percent"], 2),
            round(hrv["lf_power"], 4),
            round(hrv["hf_power"], 4)
        ])

    print("\n\n===== HRV Results =====\n")

    print(tabulate(
        results,
        headers=[
            "Video_No",
            "SDNN (ms)",
            "RMSSD (ms)",
            "pNN50 (%)",
            "LF Power",
            "HF Power"
        ],
        tablefmt="grid"
    ))


if __name__ == "__main__":
    main()