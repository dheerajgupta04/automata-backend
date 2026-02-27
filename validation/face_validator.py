import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh


class FaceValidator:
    def __init__(self):
        self.face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def validate(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False, None

        if len(results.multi_face_landmarks) > 1:
            return False, None

        return True, results.multi_face_landmarks[0]