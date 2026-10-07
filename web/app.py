import os
import sys
import json
import pickle
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.hand_tracking import HandTracker
from utils.feature_extraction import extract_features


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = os.path.join(
    ROOT, "model", "word", "gesture_model.pkl"
)

ENCODER_PATH = os.path.join(
    ROOT, "model", "word", "label_encoder.pkl"
)

MESSAGES_PATH = os.path.join(
    ROOT, "config", "gestures.json"
)

SETTINGS_PATH = os.path.join(
    ROOT, "config", "settings.json"
)


# --------------------------------------------------
# FLASK
# --------------------------------------------------

app = Flask(
    __name__,
    template_folder=os.path.join(ROOT, "web", "templates"),
    static_folder=os.path.join(ROOT, "web", "static")
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
    SETTINGS = json.load(f)

CONFIDENCE_THRESHOLD = float(
    SETTINGS.get("CONFIDENCE_THRESHOLD", 0.60)
)

# Web deployment: keep two-hand support
MAX_HANDS = 2


# --------------------------------------------------
# GESTURE MESSAGES
# --------------------------------------------------

try:
    with open(MESSAGES_PATH, "r", encoding="utf-8") as f:
        WORD_MESSAGES = json.load(f)
except Exception:
    WORD_MESSAGES = {}


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading SignSense AI web model...")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully.")

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

print("Label encoder loaded successfully.")

print(
    f"Confidence threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%"
)

print(f"Maximum hands: {MAX_HANDS}")


# --------------------------------------------------
# MEDIAPIPE TRACKER
# --------------------------------------------------

tracker = HandTracker(max_hands=MAX_HANDS)


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "label_encoder_loaded": label_encoder is not None,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "max_hands": MAX_HANDS
    })


# --------------------------------------------------
# PREDICT
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ------------------------------------------
        # CHECK IMAGE
        # ------------------------------------------

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "error": "No image received"
            }), 400

        image_file = request.files["image"]

        image_bytes = image_file.read()

        if not image_bytes:
            return jsonify({
                "success": False,
                "error": "Empty image"
            }), 400


        # ------------------------------------------
        # DECODE IMAGE
        # ------------------------------------------

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
                "error": "Could not decode image"
            }), 400


        # ------------------------------------------
        # RESIZE FOR RENDER
        # ------------------------------------------

        max_width = 320

        if frame.shape[1] > max_width:

            scale = (
                max_width /
                frame.shape[1]
            )

            new_height = int(
                frame.shape[0] * scale
            )

            frame = cv2.resize(
                frame,
                (max_width, new_height),
                interpolation=cv2.INTER_AREA
            )


        # ------------------------------------------
        # MEDIA PIPE
        # ------------------------------------------

        results = tracker.process(frame)


        # ------------------------------------------
        # NO HAND
        # ------------------------------------------

        if not results.multi_hand_landmarks:

            return jsonify({
                "success": True,
                "gesture": "-",
                "message": "No hand detected",
                "confidence": 0.0,
                "confidence_percent": 0.0,
                "hand_count": 0,
                "landmarks": [],
                "top_predictions": []
            })


        # ------------------------------------------
        # HAND COUNT
        # ------------------------------------------

        hand_count = len(
            results.multi_hand_landmarks
        )


        # ------------------------------------------
        # LANDMARKS FOR UI
        # ------------------------------------------

        landmarks = []

        for hand in results.multi_hand_landmarks:

            hand_points = []

            for landmark in hand.landmark:

                hand_points.append({
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z)
                })

            landmarks.append(hand_points)


        # ------------------------------------------
        # FEATURES
        # ------------------------------------------

        hands = [
            hand.landmark
            for hand in results.multi_hand_landmarks
        ]

        features = extract_features(
            hands,
            max_hands=MAX_HANDS
        )


        # ------------------------------------------
        # MODEL PREDICTION
        # ------------------------------------------

        probabilities = model.predict_proba(
            [features]
        )[0]


        # ------------------------------------------
        # BEST PREDICTION
        # ------------------------------------------

        best_index = int(
            np.argmax(probabilities)
        )

        confidence = float(
            probabilities[best_index]
        )

        gesture = str(
            label_encoder.inverse_transform(
                [best_index]
            )[0]
        )

        message = WORD_MESSAGES.get(
            gesture,
            gesture
        )


        # ------------------------------------------
        # TOP 3
        # ------------------------------------------

        top_indices = np.argsort(
            probabilities
        )[-3:][::-1]

        top_predictions = []

        for index in top_indices:

            label = str(
                label_encoder.inverse_transform(
                    [int(index)]
                )[0]
            )

            probability = float(
                probabilities[index]
            )

            top_predictions.append({
                "gesture": label,
                "confidence": probability,
                "confidence_percent": round(
                    probability * 100,
                    1
                )
            })


        # ------------------------------------------
        # RESPONSE
        # ------------------------------------------

        return jsonify({

            "success": True,

            "gesture": gesture,

            "message": message,

            "confidence": confidence,

            "confidence_percent": round(
                confidence * 100,
                1
            ),

            "hand_count": hand_count,

            "landmarks": landmarks,

            "top_predictions": top_predictions,

            "threshold": CONFIDENCE_THRESHOLD

        })


    except Exception as e:

        print(
            f"[PREDICT ERROR] {type(e).__name__}: {e}"
        )

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),
        debug=False
    )