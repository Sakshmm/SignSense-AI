# ==========================================================
# SignSense AI - Web Application
# Browser Camera + MediaPipe + Existing ML Model
# ==========================================================

import os
import sys
import json
import time
import pickle

import cv2
import numpy as np

from flask import Flask, render_template, request, jsonify


# ==========================================================
# PROJECT ROOT
# ==========================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ==========================================================
# EXISTING PROJECT UTILITIES
# ==========================================================

from utils.hand_tracking import HandTracker
from utils.feature_extraction import extract_features


# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = os.path.join(
    ROOT,
    "model",
    "word",
    "gesture_model.pkl"
)

ENCODER_PATH = os.path.join(
    ROOT,
    "model",
    "word",
    "label_encoder.pkl"
)

MESSAGES_PATH = os.path.join(
    ROOT,
    "config",
    "gestures.json"
)

SETTINGS_PATH = os.path.join(
    ROOT,
    "config",
    "settings.json"
)


# ==========================================================
# LOAD SETTINGS
# ==========================================================

with open(
    SETTINGS_PATH,
    "r",
    encoding="utf-8"
) as f:

    SETTINGS = json.load(f)


CONFIDENCE_THRESHOLD = float(
    SETTINGS.get(
        "CONFIDENCE_THRESHOLD",
        0.60
    )
)

# IMPORTANT:
# SignSense AI supports TWO HANDS
MAX_HANDS = 2


# ==========================================================
# LOAD GESTURE MESSAGES
# ==========================================================

try:

    with open(
        MESSAGES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        WORD_MESSAGES = json.load(f)

except Exception:

    WORD_MESSAGES = {}


# ==========================================================
# LOAD MODEL
# ==========================================================

print("=" * 60)
print("Loading SignSense AI web model...")
print("=" * 60)

with open(
    MODEL_PATH,
    "rb"
) as f:

    model = pickle.load(f)

print("Model loaded successfully.")


# ==========================================================
# LOAD LABEL ENCODER
# ==========================================================

with open(
    ENCODER_PATH,
    "rb"
) as f:

    label_encoder = pickle.load(f)

print("Label encoder loaded successfully.")

print(
    f"Confidence threshold: "
    f"{CONFIDENCE_THRESHOLD * 100:.1f}%"
)

print(
    f"Maximum hands: {MAX_HANDS}"
)


# ==========================================================
# MEDIAPIPE TRACKER
# ==========================================================

tracker = HandTracker(
    max_hands=MAX_HANDS
)


# ==========================================================
# FLASK APP
# ==========================================================

app = Flask(
    __name__,
    template_folder=os.path.join(
        ROOT,
        "web",
        "templates"
    )
)


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.route("/health")
def health():

    return jsonify({

        "status": "ok",

        "model_loaded":
            model is not None,

        "label_encoder_loaded":
            label_encoder is not None,

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "max_hands":
            MAX_HANDS
    })


# ==========================================================
# PREDICTION API
# ==========================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # --------------------------------------------------
        # CHECK IMAGE
        # --------------------------------------------------

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image received."
            }), 400


        image_file = request.files[
            "image"
        ]


        # --------------------------------------------------
        # READ IMAGE
        # --------------------------------------------------

        image_bytes = image_file.read()

        if not image_bytes:

            return jsonify({
                "success": False,
                "error": "Empty image received."
            }), 400


        # --------------------------------------------------
        # CONVERT IMAGE TO NUMPY
        # --------------------------------------------------

        np_array = np.frombuffer(
            image_bytes,
            np.uint8
        )


        frame = cv2.imdecode(
            np_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return jsonify({
                "success": False,
                "error": "Could not decode image."
            }), 400


        # --------------------------------------------------
        # RESIZE FRAME
        # REDUCES SERVER LOAD
        # --------------------------------------------------

        max_width = 480

        if frame.shape[1] > max_width:

            scale = max_width / frame.shape[1]

            new_height = int(
                frame.shape[0] * scale
            )

            frame = cv2.resize(
                frame,
                (
                    max_width,
                    new_height
                )
            )


        # --------------------------------------------------
        # MEDIAPIPE
        # TWO HANDS
        # --------------------------------------------------

        results = tracker.process(
            frame
        )


        # --------------------------------------------------
        # NO HAND
        # --------------------------------------------------

        if not results.multi_hand_landmarks:

            return jsonify({

                "success": True,

                "gesture": "-",

                "message":
                    "No hand detected",

                "confidence": 0.0,

                "confidence_percent": 0.0,

                "hand_count": 0,

                "landmarks": []

            })


        # --------------------------------------------------
        # HAND COUNT
        # --------------------------------------------------

        hand_count = len(
            results.multi_hand_landmarks
        )


        # --------------------------------------------------
        # LANDMARKS
        # --------------------------------------------------

        landmarks = []


        for hand in (
            results.multi_hand_landmarks
        ):

            points = []


            for landmark in (
                hand.landmark
            ):

                points.append({

                    "x":
                        float(
                            landmark.x
                        ),

                    "y":
                        float(
                            landmark.y
                        ),

                    "z":
                        float(
                            landmark.z
                        )

                })


            landmarks.append({
                "points": points
            })


        # --------------------------------------------------
        # EXACT SAME FEATURE EXTRACTION
        # --------------------------------------------------

        hands = [

            hand.landmark

            for hand in
            results.multi_hand_landmarks

        ]


        features = extract_features(
            hands,
            max_hands=MAX_HANDS
        )


        # --------------------------------------------------
        # MODEL PREDICTION
        # --------------------------------------------------

        probabilities = model.predict_proba(
            [features]
        )[0]


        # --------------------------------------------------
        # BEST CLASS
        # --------------------------------------------------

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


        # --------------------------------------------------
        # LABEL
        # --------------------------------------------------

        gesture = str(
            label_encoder
            .inverse_transform(
                [best_index]
            )[0]
        )


        # --------------------------------------------------
        # MESSAGE
        # --------------------------------------------------

        message = WORD_MESSAGES.get(
            gesture,
            gesture
        )


        # --------------------------------------------------
        # TOP 3 PREDICTIONS
        # --------------------------------------------------

        top_indices = np.argsort(
            probabilities
        )[-3:][::-1]


        top_predictions = []


        for index in top_indices:

            index = int(index)

            try:

                label = str(
                    label_encoder
                    .inverse_transform(
                        [index]
                    )[0]
                )

            except Exception:

                label = f"class_{index}"


            top_predictions.append({

                "gesture": label,

                "confidence":
                    float(
                        probabilities[index]
                    )

            })


        # --------------------------------------------------
        # SERVER LOG
        # --------------------------------------------------

        print(
            f"[PREDICT] "
            f"Hands={hand_count} | "
            f"Gesture={gesture} | "
            f"Confidence="
            f"{confidence * 100:.1f}%"
        )


        # --------------------------------------------------
        # RESPONSE
        # --------------------------------------------------

        return jsonify({

            "success": True,

            "gesture":
                gesture,

            "message":
                message,

            "confidence":
                confidence,

            "confidence_percent":
                round(
                    confidence * 100,
                    2
                ),

            "threshold":
                CONFIDENCE_THRESHOLD,

            "hand_count":
                hand_count,

            "landmarks":
                landmarks,

            "top_predictions":
                top_predictions

        })


    except Exception as e:

        print(
            "[PREDICT ERROR]",
            str(e)
        )

        return jsonify({

            "success": False,

            "error":
                str(e)

        }), 500


# ==========================================================
# START SERVER
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("             SignSense AI - Web Version")
    print("=" * 60)
    print()

    print("Open in browser:")
    print("http://127.0.0.1:5000")
    print()

    print("Health check:")
    print("http://127.0.0.1:5000/health")
    print()

    print("Prediction API:")
    print("POST http://127.0.0.1:5000/predict")
    print()

    print("=" * 60)


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )