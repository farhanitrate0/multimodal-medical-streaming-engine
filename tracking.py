"""Face mesh + hand landmarks, with a zoom-in fallback for small or distant faces."""
import threading

import cv2
import mediapipe as mp
import numpy as np

ZOOM = 256  # crop size fed to the mesh


class SpatialTracker:
    def __init__(self, video=False):
        static = not video
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static, max_num_faces=1, refine_landmarks=True)
        self.zoom_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)  # full range
        self.hands = mp.solutions.hands.Hands(static_image_mode=static, max_num_hands=2, min_detection_confidence=0.5,
                                              model_complexity=0 if video else 1)  # lite model for live video
        self.lock = threading.Lock()  # mediapipe graphs aren't thread-safe

    def _zoomed_face(self, rgb):
        """Find a small face with the full-range detector, then run the mesh on an upscaled crop."""
        det = self.detector.process(rgb)
        if not det.detections:
            return []
        h, w = rgb.shape[:2]
        b = det.detections[0].location_data.relative_bounding_box
        side = int(1.8 * max(b.width * w, b.height * h))
        if side < 8:
            return []
        x0 = int((b.xmin + b.width / 2) * w - side / 2)
        y0 = int((b.ymin + b.height / 2) * h - side / 2)
        pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
        padded = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        crop = padded[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
        res = self.zoom_mesh.process(cv2.resize(crop, (ZOOM, ZOOM), interpolation=cv2.INTER_CUBIC))
        if not res.multi_face_landmarks:
            return []
        return [[(x0 + lm.x * side) / w, (y0 + lm.y * side) / h, lm.z * side / w]
                for lm in res.multi_face_landmarks[0].landmark]

    def process(self, rgb):
        """Normalized [x, y, z] face points and a list of 21-point hands."""
        rgb = np.ascontiguousarray(rgb)
        with self.lock:
            res = self.mesh.process(rgb)
            if res.multi_face_landmarks:
                face = [[lm.x, lm.y, lm.z] for lm in res.multi_face_landmarks[0].landmark]
            else:
                face = self._zoomed_face(rgb)
            hand_res = self.hands.process(rgb)
        hands = [[[lm.x, lm.y, lm.z] for lm in hand.landmark] for hand in (hand_res.multi_hand_landmarks or [])]
        return face, hands


def draw(bgr, face, hands, face_step=3):
    """Landmarks on a copy-safe image (bgr, in place)."""
    h, w = bgr.shape[:2]
    r = max(1, round(min(h, w) / 400))
    for x, y, _ in face[::face_step]:
        cv2.circle(bgr, (int(x * w), int(y * h)), r, (0, 200, 0), -1)
    for hand in hands:
        for x, y, _ in hand:
            cv2.circle(bgr, (int(x * w), int(y * h)), 3 * r, (255, 140, 0), -1)
    return bgr
