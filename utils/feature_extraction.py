import numpy as np


def extract_single_hand(hand_landmarks):
    """
    Convert one hand's 21 MediaPipe landmarks into
    63 normalized features.
    """

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
        dtype=np.float32
    )

    # Make wrist the origin
    wrist = points[0].copy()
    points = points - wrist

    # Scale according to hand size
    distances = np.linalg.norm(points, axis=1)
    scale = float(np.max(distances))

    if scale > 1e-6:
        points = points / scale

    return points.flatten().astype(np.float32)


def extract_features(hand_landmarks_list, max_hands=2):
    """
    Supports 1 or 2 hands.

    1 hand:
        63 features

    2 hands:
        126 features

    If only one hand is visible, the second-hand
    features are filled with zeros.
    """

    features = []

    # Process available hands
    for hand in hand_landmarks_list[:max_hands]:
        features.extend(
            extract_single_hand(hand)
        )

    # Add zeros if fewer hands are detected
    while len(features) < max_hands * 63:
        features.extend([0.0] * 63)

    return np.array(
        features[:max_hands * 63],
        dtype=np.float32
    )
