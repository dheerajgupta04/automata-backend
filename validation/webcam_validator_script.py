import cv2
import numpy as np
import os
import time
from validation.face_validator import FaceValidator
from validation.brightness_validator import validate_brightness
from config import *

def main():
    # Initialize face validator
    face_validator = FaceValidator()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, MIN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MIN_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, MIN_FPS)

    # Output directory
    temp_dir = os.path.join(os.getcwd(), "temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    print(f"Starting validation. Need {TOTAL_FRAMES} valid frames ({MIN_FPS} FPS, {MIN_DURATION}s).")
    print(f"Thresholds: Brightness:[{DARK_THRESHOLD}-{BRIGHT_THRESHOLD}], Green:[{GREEN_LOW}-{GREEN_HIGH}], StdDev:[{BRIGHTNESS_STD - 4}-{BRIGHTNESS_STD + 4}], Contrast>{MIN_CONTRAST}")

    valid_frames = []
    
    try:
        while len(valid_frames) < TOTAL_FRAMES:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # 1. Face Validation
            face_valid, _ = face_validator.validate(frame)
            if not face_valid:
                print(f"Frame {len(valid_frames)}: Face detection failed. Resetting.", end='\r')
                valid_frames = [] 
                cv2.imshow('Validation', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # 2. Brightness/Color/Contrast Validation
            b_result = validate_brightness(frame)
            
            # Additional check for Std Dev and Stable Brightness Range
            # Std Dev Range: 10 - 18 (from user request)
            is_std_valid = 10 <= b_result["std"] <= 18
            # Brightness Mean: 115 - 130
            is_mean_valid = 115 <= b_result["mean"] <= 130
            # Green Mean: 110 - 130
            is_green_valid = 110 <= b_result["green_mean"] <= 130
            # Contrast > 50
            is_contrast_valid = b_result["contrast"] > 50
            
            if is_mean_valid and is_green_valid and is_std_valid and is_contrast_valid:
                valid_frames.append(frame)
                status = f"Valid: {len(valid_frames)}/{TOTAL_FRAMES} | Mean: {b_result['mean']:.1f} | Green: {b_result['green_mean']:.1f} | Std: {b_result['std']:.1f}"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                valid_frames = [] # Reset on any failure
                reasons = []
                if not is_mean_valid: reasons.append("Mean")
                if not is_green_valid: reasons.append("Green")
                if not is_std_valid: reasons.append("StdDev")
                if not is_contrast_valid: reasons.append("Contrast")
                
                reason_str = ", ".join(reasons)
                print(f"Rejected: {reason_str}. Resetting counter.             ", end='\r')
                cv2.putText(frame, f"INVALID: {reason_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('Validation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if len(valid_frames) == TOTAL_FRAMES:
            print("\nSuccessfully collected valid video (30s). Saving...")
            
            timestamp = int(time.time())
            filename = os.path.join(temp_dir, f"validated_video_{timestamp}.mp4")
            
            height, width, _ = valid_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, MIN_FPS, (width, height))
            
            for f in valid_frames:
                out.write(f)
            out.release()
            
            print(f"Video saved: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
