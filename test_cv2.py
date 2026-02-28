import cv2
import sys

def test_video(path):
    print(f"Testing video: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Failed to open video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"FPS: {fps}")
    print(f"Frame Count: {frame_count}")
    
    success, img = cap.read()
    if success:
        print(f"First frame shape: {img.shape}")
    else:
        print("Failed to read first frame")
        
    cap.release()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_video(sys.argv[1])
    else:
        print("Please provide video path")
