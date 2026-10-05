import csv
from pathlib import Path

import cv2

from utils.hand_tracking import HandTracker
from utils.feature_extraction import extract_features


ROOT = Path(__file__).resolve().parent

DATASET_FOLDER = ROOT / "dataset" / "word"

CAMERA_INDEX = 0

SAMPLES_PER_GESTURE = 300

MAX_HANDS = 2

TOTAL_FEATURES = 126


def collect_gesture(label):

    DATASET_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        DATASET_FOLDER /
        f"{label}.csv"
    )

    # ------------------------------------------------
    # IMPORTANT:
    # Don't overwrite an existing CSV.
    # ------------------------------------------------

    if csv_path.exists():

        print()
        print(
            f"{label}.csv already exists!"
        )

        choice = input(
            "Overwrite it? [y/N]: "
        ).strip().lower()

        if choice != "y":

            print(
                "Collection cancelled."
            )

            return

    tracker = HandTracker(
        max_hands=MAX_HANDS
    )

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        print(
            "ERROR: Could not open webcam."
        )

        tracker.close()

        return

    # ------------------------------------------------
    # CSV HEADER
    # ------------------------------------------------

    header = [
        f"f{i}"
        for i in range(TOTAL_FEATURES)
    ]

    header.append("label")

    count = 0

    recording = False

    print()
    print("=" * 60)
    print(f"Collecting gesture: {label}")
    print("=" * 60)

    print()
    print("SPACE = Start / Pause recording")
    print("ESC   = Stop")
    print()
    print(
        "You can use 1 hand OR 2 hands."
    )
    print(
        "The system will detect them automatically."
    )
    print()

    # ------------------------------------------------
    # CREATE CSV
    # ------------------------------------------------

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)

        # ------------------------------------------------
        # CAMERA LOOP
        # ------------------------------------------------

        while True:

            success, frame = camera.read()

            if not success:

                print(
                    "Could not read webcam frame."
                )

                break

            # Mirror camera
            frame = cv2.flip(
                frame,
                1
            )

            # MediaPipe
            results = tracker.process(
                frame
            )

            # Draw ALL detected hands
            frame, boxes = tracker.draw(
                frame,
                results
            )

            # ------------------------------------------------
            # RECORD SAMPLE
            # ------------------------------------------------

            if recording:

                if results.multi_hand_landmarks:

                    hands = [
                        hand.landmark
                        for hand
                        in results.multi_hand_landmarks
                    ]

                    # Automatically handles:
                    #
                    # 1 hand:
                    # 63 real features
                    # + 63 zeros
                    #
                    # 2 hands:
                    # 63 + 63
                    #
                    features = extract_features(
                        hands,
                        max_hands=MAX_HANDS
                    )

                    writer.writerow(
                        features.tolist()
                        + [label]
                    )

                    count += 1

                    if count >= SAMPLES_PER_GESTURE:

                        break

            # ------------------------------------------------
            # UI
            # ------------------------------------------------

            status = (
                "RECORDING"
                if recording
                else "PAUSED"
            )

            hands_detected = 0

            if results.multi_hand_landmarks:

                hands_detected = len(
                    results.multi_hand_landmarks
                )

            cv2.putText(
                frame,
                f"Gesture: {label}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"{status}  "
                f"{count}/{SAMPLES_PER_GESTURE}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    (0, 255, 0)
                    if recording
                    else (0, 200, 255)
                ),
                2
            )

            cv2.putText(
                frame,
                f"Hands detected: "
                f"{hands_detected}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "SPACE = Record/Pause | ESC = Exit",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "SignSense AI - Dataset Collection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # ESC
            if key == 27:

                break

            # SPACE
            if key == 32:

                recording = not recording

    # ------------------------------------------------
    # CLEANUP
    # ------------------------------------------------

    camera.release()

    tracker.close()

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("COLLECTION FINISHED")
    print("=" * 60)

    print(
        f"Samples collected: {count}"
    )

    print(
        f"Saved file: {csv_path}"
    )


# ====================================================
# MAIN
# ====================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SignSense AI - Patient Gesture Dataset")
    print("=" * 60)

    print()
    print("Existing gestures are preserved.")
    print("You can now add new gestures.")
    print()

    while True:

        label = input(
            "Enter new gesture label "
            "(or Q to quit): "
        ).strip().upper()

        if label == "Q":

            break

        if not label:

            print(
                "Label cannot be empty."
            )

            continue

        collect_gesture(
            label
        )

        print()

        again = input(
            "Collect another gesture? [Y/n]: "
        ).strip().lower()

        if again == "n":

            break
