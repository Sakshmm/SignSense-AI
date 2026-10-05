import json
import time
import threading
from pathlib import Path

import cv2
import pickle
import numpy as np

from utils.hand_tracking import HandTracker
from utils.feature_extraction import extract_features
from utils.gesture_stability import GestureStability
from utils.speech import SpeechManager
from utils.history import SessionHistory


ROOT = Path(__file__).resolve().parent


# =========================================================
# CONFIG
# =========================================================

with open(
    ROOT / "config" / "gestures.json",
    "r",
    encoding="utf-8"
) as f:
    WORD_MESSAGES = json.load(f)

with open(
    ROOT / "config" / "settings.json",
    "r",
    encoding="utf-8"
) as f:
    SETTINGS = json.load(f)


# =========================================================
# FAST RESPONSE SETTINGS
# =========================================================
#
# Confidence is between 0.0 and 1.0
#
# 0.60 = 60%
#
# 3 frames at ~30 FPS ≈ 0.1 second
# 0.8 second cooldown prevents repeated speech spam
# =========================================================

CONFIDENCE_THRESHOLD = 0.60

STABLE_FRAMES = 3

COOLDOWN_SECONDS = 0.8


# =========================================================
# OTHER SETTINGS
# =========================================================

CAMERA_INDEX = int(
    SETTINGS["CAMERA_INDEX"]
)

MAX_HANDS = int(
    SETTINGS["MAX_HANDS"]
)


# =========================================================
# MODEL
# =========================================================

MODEL_PATH = (
    ROOT
    / "model"
    / "word"
    / "gesture_model.pkl"
)

ENCODER_PATH = (
    ROOT
    / "model"
    / "word"
    / "label_encoder.pkl"
)


if not MODEL_PATH.exists():

    raise SystemExit(
        "Trained model not found.\n"
        "Run train_model.py first."
    )


if not ENCODER_PATH.exists():

    raise SystemExit(
        "Label encoder not found.\n"
        "Run train_model.py first."
    )


with open(
    MODEL_PATH,
    "rb"
) as f:

    model = pickle.load(f)


with open(
    ENCODER_PATH,
    "rb"
) as f:

    label_encoder = pickle.load(f)


# =========================================================
# COMPONENTS
# =========================================================

tracker = HandTracker(
    max_hands=MAX_HANDS
)


stability = GestureStability(
    stable_frames=STABLE_FRAMES,
    cooldown=COOLDOWN_SECONDS
)


speech = SpeechManager()


history = SessionHistory()


# =========================================================
# DISPLAY VARIABLES
# =========================================================

last_gesture = "-"

last_confidence = 0.0

last_message = "Waiting..."

speaking_until = 0.0

emergency_until = 0.0


# =========================================================
# BEEP
# =========================================================

def beep():

    try:

        import winsound

        winsound.Beep(
            1000,
            500
        )

    except Exception:

        print(
            "\a",
            end="",
            flush=True
        )


# =========================================================
# SPEAK
# =========================================================

def speak(text):

    global speaking_until

    if text:

        # UI indicator
        speaking_until = (
            time.time() + 2.0
        )

        # Speech happens in background
        speech.speak(text)


# =========================================================
# GESTURE TRIGGER
# =========================================================

def handle_gesture(
    gesture,
    confidence
):

    global last_message
    global emergency_until

    # -----------------------------------------------------
    # Get custom message
    # -----------------------------------------------------

    message = WORD_MESSAGES.get(
        gesture,
        gesture
    )

    last_message = message


    # -----------------------------------------------------
    # Save history
    # -----------------------------------------------------

    history.add(
        "Patient",
        gesture,
        message,
        confidence
    )


    # -----------------------------------------------------
    # Speak
    # -----------------------------------------------------

    speak(message)


    # -----------------------------------------------------
    # Emergency gestures
    # -----------------------------------------------------

    if gesture in {
        "HELP",
        "PAIN",
        "SOS",
        "SEVERE_PAIN",
        "EMERGENCY",
        "CALL_NURSE",
        "CALL_DOCTOR"
    }:

        emergency_until = (
            time.time() + 4.0
        )

        threading.Thread(
            target=beep,
            daemon=True
        ).start()


# =========================================================
# DRAW TEXT
# =========================================================

def draw_text(
    frame,
    text,
    x,
    y,
    scale=0.6,
    color=(255, 255, 255),
    thickness=1
):

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# =========================================================
# UI
# =========================================================

def draw_ui(frame):

    h, w = frame.shape[:2]


    # -----------------------------------------------------
    # TOP BAR
    # -----------------------------------------------------

    cv2.rectangle(
        frame,
        (0, 0),
        (w, 125),
        (25, 25, 25),
        -1
    )


    draw_text(
        frame,
        "SignSense AI",
        20,
        32,
        0.9,
        (0, 255, 255),
        2
    )


    draw_text(
        frame,
        "Patient Communication Mode",
        20,
        65,
        0.6,
        (255, 255, 255),
        2
    )


    draw_text(
        frame,
        "Detected:",
        20,
        98,
        0.6,
        (200, 200, 200),
        1
    )


    draw_text(
        frame,
        last_gesture,
        115,
        98,
        0.65,
        (0, 255, 0),
        2
    )


    # -----------------------------------------------------
    # BOTTOM PANEL
    # -----------------------------------------------------

    panel_y = h - 190


    cv2.rectangle(
        frame,
        (0, panel_y),
        (w, h),
        (20, 20, 20),
        -1
    )


    draw_text(
        frame,
        f"Confidence: "
        f"{last_confidence * 100:.1f}%",
        20,
        panel_y + 30,
        0.65
    )


    # -----------------------------------------------------
    # LOW CONFIDENCE
    # -----------------------------------------------------

    if (
        last_confidence
        < CONFIDENCE_THRESHOLD
        and last_gesture != "-"
    ):

        draw_text(
            frame,
            "Low confidence - "
            "show gesture clearly",
            20,
            panel_y + 60,
            0.6,
            (0, 200, 255),
            2
        )

    else:

        draw_text(
            frame,
            f"Message: {last_message}",
            20,
            panel_y + 60,
            0.6
        )


    # -----------------------------------------------------
    # CONTROLS
    # -----------------------------------------------------

    draw_text(
        frame,
        "ESC = Exit",
        20,
        panel_y + 95,
        0.55,
        (200, 200, 200),
        1
    )


    # -----------------------------------------------------
    # HISTORY
    # -----------------------------------------------------

    draw_text(
        frame,
        "Session History:",
        int(w * 0.58),
        panel_y + 30,
        0.6,
        (0, 255, 255),
        2
    )


    y = panel_y + 58


    for item in history.recent()[:5]:

        line = (
            f"{item['time']} "
            f"{item['gesture']} "
            f"{item['confidence'] * 100:.0f}%"
        )


        draw_text(
            frame,
            line,
            int(w * 0.58),
            y,
            0.45
        )


        y += 22


    # -----------------------------------------------------
    # EMERGENCY
    # -----------------------------------------------------

    if time.time() < emergency_until:

        cv2.rectangle(
            frame,
            (0, 125),
            (w, panel_y),
            (0, 0, 255),
            5
        )


        draw_text(
            frame,
            "!!! EMERGENCY !!!",
            int(w * 0.32),
            160,
            0.9,
            (0, 0, 255),
            3
        )


    # -----------------------------------------------------
    # SPEAKING
    # -----------------------------------------------------

    if time.time() < speaking_until:

        draw_text(
            frame,
            "Speaking...",
            int(w * 0.42),
            195,
            0.65,
            (0, 255, 255),
            2
        )


# =========================================================
# MAIN
# =========================================================

def main():

    global last_gesture
    global last_confidence
    global last_message


    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )


    if not camera.isOpened():

        raise SystemExit(
            "Could not open webcam."
        )


    print()
    print("=" * 55)
    print("SignSense AI Started")
    print("=" * 55)
    print()

    print(
        "Fast Response Mode Enabled"
    )

    print(
        "Confidence Threshold: 60%"
    )

    print(
        "Stable Frames: 3"
    )

    print(
        "Cooldown: 0.8 seconds"
    )

    print()

    print(
        "1 or 2 hands are detected automatically."
    )

    print(
        "ESC = Exit"
    )

    print()


    try:

        while True:

            # -------------------------------------------------
            # READ CAMERA
            # -------------------------------------------------

            success, frame = camera.read()


            if not success:

                print(
                    "Could not read webcam frame."
                )

                break


            # -------------------------------------------------
            # MIRROR CAMERA
            # -------------------------------------------------

            frame = cv2.flip(
                frame,
                1
            )


            # -------------------------------------------------
            # MEDIAPIPE
            # -------------------------------------------------

            results = tracker.process(
                frame
            )


            frame, _ = tracker.draw(
                frame,
                results
            )


            # =================================================
            # HAND DETECTED
            # =================================================

            if results.multi_hand_landmarks:


                # -------------------------------------------------
                # Collect all hands
                # -------------------------------------------------

                hands = [
                    hand.landmark
                    for hand
                    in results.multi_hand_landmarks
                ]


                # -------------------------------------------------
                # Extract 126 features
                # -------------------------------------------------

                features = extract_features(
                    hands,
                    max_hands=MAX_HANDS
                )


                # -------------------------------------------------
                # MODEL PREDICTION
                # -------------------------------------------------

                probabilities = (
                    model.predict_proba(
                        [features]
                    )[0]
                )


                best_index = int(
                    np.argmax(
                        probabilities
                    )
                )


                confidence = float(
                    probabilities[
                        best_index
                    ]
                )


                gesture = str(
                    label_encoder.inverse_transform(
                        [best_index]
                    )[0]
                )


                # -------------------------------------------------
                # UPDATE UI
                # -------------------------------------------------

                last_gesture = gesture

                last_confidence = confidence


                # -------------------------------------------------
                # CONFIDENCE + STABILITY
                # -------------------------------------------------

                if (
                    confidence
                    >= CONFIDENCE_THRESHOLD
                ):


                    triggered = (
                        stability.update(
                            gesture,
                            confidence
                        )
                    )


                    # -------------------------------------------------
                    # GESTURE CONFIRMED
                    # -------------------------------------------------

                    if triggered is not None:

                        handle_gesture(
                            triggered,
                            confidence
                        )


                else:

                    stability.update(
                        None,
                        confidence
                    )


                    last_message = (
                        "Low confidence - "
                        "show gesture clearly"
                    )


            # =================================================
            # NO HAND
            # =================================================

            else:

                last_gesture = "-"

                last_confidence = 0.0


                stability.update(
                    None,
                    0.0
                )


                last_message = (
                    "No hand detected"
                )


            # =================================================
            # DRAW UI
            # =================================================

            draw_ui(
                frame
            )


            # -------------------------------------------------
            # SHOW WINDOW
            # -------------------------------------------------

            cv2.imshow(
                "SignSense AI",
                frame
            )


            # -------------------------------------------------
            # KEYBOARD
            # -------------------------------------------------

            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            if key == 27:

                break


    finally:

        # -------------------------------------------------
        # CLEANUP
        # -------------------------------------------------

        camera.release()

        tracker.close()

        speech.stop()

        cv2.destroyAllWindows()


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()