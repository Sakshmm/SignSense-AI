import csv
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parent

DATASET_FOLDER = ROOT / "dataset" / "word"
MODEL_FOLDER = ROOT / "model" / "word"

# Final model always uses 126 features:
# Hand 1 = 63
# Hand 2 = 63
TOTAL_FEATURES = 126


def train_model():

    MODEL_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_files = sorted(
        DATASET_FOLDER.glob("*.csv")
    )

    if not csv_files:
        print("No gesture CSV files found.")
        return

    rows = []

    print()
    print("=" * 60)
    print("READING GESTURE DATA")
    print("=" * 60)

    for csv_file in csv_files:

        with open(
            csv_file,
            "r",
            newline="",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                try:

                    # Read existing 63-feature data
                    features = [
                        float(row[f"f{i}"])
                        for i in range(63)
                    ]

                    label = row["label"].strip()

                    if not label:
                        continue

                    # ------------------------------------------------
                    # IMPORTANT
                    # Existing data = 1 hand = 63 features
                    #
                    # Add 63 zeros for the missing second hand.
                    #
                    # 63 + 63 = 126
                    # ------------------------------------------------

                    features += [0.0] * 63

                    rows.append(
                        features + [label]
                    )

                except (
                    KeyError,
                    ValueError
                ):
                    continue

    if not rows:
        print("No valid training data found.")
        return

    # ------------------------------------------------
    # X = features
    # Y = labels
    # ------------------------------------------------

    X = np.array(
        [
            row[:TOTAL_FEATURES]
            for row in rows
        ],
        dtype=np.float32
    )

    y_text = np.array(
        [
            row[TOTAL_FEATURES]
            for row in rows
        ]
    )

    # ------------------------------------------------
    # LABEL ENCODER
    # ------------------------------------------------

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        y_text
    )

    print()
    print(f"Total samples: {len(X)}")
    print(
        f"Total gestures: "
        f"{len(label_encoder.classes_)}"
    )

    print()
    print("Gestures:")

    for label in label_encoder.classes_:
        print(" -", label)

    # ------------------------------------------------
    # NEED AT LEAST 2 GESTURES
    # ------------------------------------------------

    if len(label_encoder.classes_) < 2:

        print()
        print(
            "At least 2 gestures are required."
        )

        return

    # ------------------------------------------------
    # TRAIN / TEST SPLIT
    # ------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------------
    # RANDOM FOREST
    # ------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print()
    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    # ------------------------------------------------
    # TEST MODEL
    # ------------------------------------------------

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()
    print("Classification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(
                len(label_encoder.classes_)
            ),
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    # ------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------

    model_path = (
        MODEL_FOLDER /
        "gesture_model.pkl"
    )

    encoder_path = (
        MODEL_FOLDER /
        "label_encoder.pkl"
    )

    info_path = (
        MODEL_FOLDER /
        "model_info.json"
    )

    with open(
        model_path,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )

    with open(
        encoder_path,
        "wb"
    ) as file:

        pickle.dump(
            label_encoder,
            file
        )

    model_info = {

        "model_type":
            "RandomForestClassifier",

        "features_per_hand":
            63,

        "total_features":
            126,

        "gestures":
            label_encoder.classes_.tolist(),

        "samples":
            len(X),

        "accuracy":
            float(accuracy),

        "supports_one_hand":
            True,

        "supports_two_hands":
            True
    }

    with open(
        info_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            model_info,
            file,
            indent=4
        )

    print()
    print("Model saved successfully:")
    print(model_path)

    print()
    print("Encoder saved:")
    print(encoder_path)

    print()
    print("Model information saved:")
    print(info_path)


if __name__ == "__main__":
    train_model()
