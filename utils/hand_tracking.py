import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


class HandTracker:
    """MediaPipe Hands wrapper supporting 1 or 2 hands."""

    def __init__(self, max_hands=1):
        self.max_hands = max_hands

        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame):
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        return self.hands.process(rgb)

    def draw(self, frame, results):
        if not results.multi_hand_landmarks:
            return frame, []

        boxes = []

        # Draw ALL detected hands
        for hand in results.multi_hand_landmarks:

            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

            # Get hand points
            h, w = frame.shape[:2]

            points = np.array(
                [
                    (int(lm.x * w), int(lm.y * h))
                    for lm in hand.landmark
                ],
                dtype=np.int32
            )

            # Bounding box
            x, y, bw, bh = cv2.boundingRect(points)

            pad = 20

            x1 = max(0, x - pad)
            y1 = max(0, y - pad)

            x2 = min(w - 1, x + bw + pad)
            y2 = min(h - 1, y + bh + pad)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            boxes.append(
                (x1, y1, x2, y2)
            )

        return frame, boxes

    def close(self):
        self.hands.close()
