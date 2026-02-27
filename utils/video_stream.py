import cv2

def get_stream(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise Exception("Cannot open video source")
    return cap