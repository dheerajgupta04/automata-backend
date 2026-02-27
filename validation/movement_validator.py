import numpy as np

class MovementValidator:
    def __init__(self):
        self.prev_points = None

    def validate(self, landmarks):
        points = [(lm.x, lm.y) for lm in landmarks.landmark]
        points = np.array(points)

        if self.prev_points is None:
            self.prev_points = points
            return True

        movement = np.mean(np.linalg.norm(points - self.prev_points, axis=1))
        self.prev_points = points

        return movement < 0.01