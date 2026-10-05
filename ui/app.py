import sys
import json
import base64
import threading

from pathlib import Path

import cv2

from PySide6.QtCore import (
    QObject,
    QThread,
    Signal,
    Slot,
    QTimer,
    QBuffer,
    QIODevice
)

from PySide6.QtWidgets import QApplication

from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(ROOT)
)


# ============================================================
# IMPORTANT:
# IMPORT YOUR ORIGINAL WORKING MAIN.PY
#
# We are NOT rewriting the ML pipeline.
# We reuse the exact model/tracker/stability/speech/history
# objects and functions from main.py.
# ============================================================

import main as core


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = core.CAMERA_INDEX


# ============================================================
# SHARED FRAME
# ============================================================

class SharedFrame:

    def __init__(self):

        self.lock = threading.Lock()

        self.frame = None


    def set(self, frame):

        with self.lock:

            self.frame = frame.copy()


    def get(self):

        with self.lock:

            if self.frame is None:

                return None

            return self.frame.copy()


# ============================================================
# CAMERA WORKER
# ============================================================

class CameraWorker(QObject):

    frameReady = Signal(str)

    cameraReady = Signal()

    error = Signal(str)

    finished = Signal()


    def __init__(
        self,
        shared_frame
    ):

        super().__init__()

        self.shared_frame = shared_frame

        self.running = False

        self.camera = None

        self.frame_count = 0


    # ========================================================
    # START
    # ========================================================

    @Slot()
    def start(self):

        try:

            self.camera = cv2.VideoCapture(
                CAMERA_INDEX
            )


            if not self.camera.isOpened():

                raise RuntimeError(
                    "Could not open webcam."
                )


            # Same camera settings
            # but small resolution for smooth UI.

            self.camera.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                640
            )

            self.camera.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                360
            )

            self.camera.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1
            )


            self.running = True

            self.cameraReady.emit()


            self.capture_loop()


        except Exception as e:

            self.error.emit(
                str(e)
            )


    # ========================================================
    # CAMERA LOOP
    # ========================================================

    def capture_loop(self):

        if not self.running:

            self.cleanup()

            return


        try:

            success, frame = (
                self.camera.read()
            )


            if not success:

                QTimer.singleShot(
                    5,
                    self.capture_loop
                )

                return


            # Mirror camera

            frame = cv2.flip(
                frame,
                1
            )


            # Give latest frame to ML worker

            self.shared_frame.set(
                frame
            )


            self.frame_count += 1


            # Send camera to browser
            # every second frame for smoothness.

            if self.frame_count % 2 == 0:

                self.send_frame(
                    frame
                )


        except Exception as e:

            print(
                "Camera error:",
                e
            )


        if self.running:

            QTimer.singleShot(
                5,
                self.capture_loop
            )

        else:

            self.cleanup()


    # ========================================================
    # SEND FRAME TO HTML
    # ========================================================

    def send_frame(
        self,
        frame
    ):

        try:

            frame = cv2.resize(
                frame,
                (640, 360)
            )


            success, encoded = (
                cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        65
                    ]
                )
            )


            if not success:

                return


            jpg_bytes = (
                encoded.tobytes()
            )


            image_data = (
                base64.b64encode(
                    jpg_bytes
                ).decode(
                    "utf-8"
                )
            )


            self.frameReady.emit(
                image_data
            )


        except Exception as e:

            print(
                "Frame encoding error:",
                e
            )


    # ========================================================
    # STOP
    # ========================================================

    @Slot()
    def stop(self):

        self.running = False


    # ========================================================
    # CLEANUP
    # ========================================================

    def cleanup(self):

        try:

            if self.camera:

                self.camera.release()

        except Exception:

            pass


        self.finished.emit()


# ============================================================
# ML WORKER
# ============================================================

class MLWorker(QObject):

    detectionReady = Signal(
        str,
        str,
        float,
        int
    )

    landmarksReady = Signal(
        list
    )

    voiceStarted = Signal()

    voiceReady = Signal()

    emergencyChanged = Signal(
        bool
    )

    error = Signal(
        str
    )

    finished = Signal()


    def __init__(
        self,
        shared_frame
    ):

        super().__init__()

        self.shared_frame = (
            shared_frame
        )

        self.running = False

        self.last_processed_frame = None
        self.diagnostic_counter = 0


    # ========================================================
    # START
    # ========================================================

    @Slot()
    def start(self):

        try:

            # IMPORTANT:
            # Everything below comes from ORIGINAL main.py.

            if core.model is None:

                raise RuntimeError(
                    "Original trained model is not loaded."
                )


            self.running = True


            self.process_loop()


        except Exception as e:

            self.error.emit(
                str(e)
            )


    # ========================================================
    # ML LOOP
    # ========================================================

    def process_loop(self):

        if not self.running:

            self.finished.emit()

            return


        try:

            frame = (
                self.shared_frame.get()
            )


            if frame is not None:

                self.process_frame(
                    frame
                )


        except Exception as e:

            print(
                "ML error:",
                e
            )


        if self.running:

            QTimer.singleShot(
                20,
                self.process_loop
            )


    # ========================================================
    # PROCESS FRAME
    #
    # This follows the ORIGINAL main.py pipeline:
    #
    # tracker.process()
    #       ↓
    # extract_features()
    #       ↓
    # model.predict_proba()
    #       ↓
    # label_encoder
    #       ↓
    # stability
    #       ↓
    # handle_gesture()
    #       ↓
    # speech/history/emergency
    # ========================================================

    def process_frame(
        self,
        frame
    ):

        # ----------------------------------------------------
        # MEDIAPIPE
        # ----------------------------------------------------

        results = core.tracker.process(
            frame
        )


        # ====================================================
        # HAND DETECTED
        # ====================================================

        if results.multi_hand_landmarks:

            hand_count = len(
                results.multi_hand_landmarks
            )


            # ------------------------------------------------
            # LANDMARK DATA FOR FRONTEND
            # ------------------------------------------------

            hands_data = []


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
                            )

                    })


                hands_data.append({

                    "points":
                        points

                })


            # Send points to canvas

            self.landmarksReady.emit(
                hands_data
            )


            # ------------------------------------------------
            # EXACT FEATURE EXTRACTION FROM MAIN.PY
            # ------------------------------------------------

            hands = [

                hand.landmark

                for hand in
                results.multi_hand_landmarks

            ]


            features = (
                core.extract_features(
                    hands,
                    max_hands=core.MAX_HANDS
                )
            )


            # ------------------------------------------------
            # EXACT TRAINED MODEL
            # ------------------------------------------------

            probabilities = (
                core.model.predict_proba(
                    [features]
                )[0]
            )


            best_index = int(
                core.np.argmax(
                    probabilities
                )
            )


            confidence = float(
                probabilities[
                    best_index
                ]
            )

            # ------------------------------------------------
            # STEP 2 DIAGNOSTIC: TOP-3 MODEL PREDICTIONS
            # ------------------------------------------------
            # This does NOT change the prediction, threshold,
            # stability, model, or dataset. It only prints
            # what the trained model is actually seeing.
            # ------------------------------------------------
            self.diagnostic_counter = getattr(
                self,
                "diagnostic_counter",
                0
            ) + 1

            if self.diagnostic_counter % 15 == 0:
                top_indices = core.np.argsort(
                    probabilities
                )[-3:][::-1]

                diagnostic_items = []

                for idx in top_indices:
                    try:
                        label = str(
                            core.label_encoder
                            .inverse_transform([int(idx)])[0]
                        )
                    except Exception:
                        label = f"class_{int(idx)}"

                    diagnostic_items.append(
                        f"{label}={float(probabilities[int(idx)]) * 100:.1f}%"
                    )

                print(
                    "\n[STEP 2 DIAGNOSTIC] "
                    f"Hands={hand_count} | "
                    f"Top-3: {' | '.join(diagnostic_items)} | "
                    f"Threshold={core.CONFIDENCE_THRESHOLD * 100:.1f}%"
                )


            gesture = str(
                core.label_encoder
                .inverse_transform(
                    [best_index]
                )[0]
            )


            # ------------------------------------------------
            # MESSAGE
            # ------------------------------------------------

            message = (
                core.WORD_MESSAGES.get(
                    gesture,
                    gesture
                )
            )


            # ------------------------------------------------
            # SEND CURRENT PREDICTION
            # ------------------------------------------------

            self.detectionReady.emit(

                gesture,

                message,

                confidence,

                hand_count

            )


            # ------------------------------------------------
            # EXACT STABILITY LOGIC
            # ------------------------------------------------

            if (
                confidence >=
                core.CONFIDENCE_THRESHOLD
            ):

                triggered = (
                    core.stability.update(
                        gesture,
                        confidence
                    )
                )


                if triggered is not None:

                    # Use original main.py
                    # handle_gesture()
                    #
                    # This automatically:
                    # - saves history
                    # - speaks message
                    # - handles emergency

                    before_speaking = (
                        core.speaking_until
                    )


                    core.handle_gesture(
                        triggered,
                        confidence
                    )


                    # Voice started

                    if (
                        core.speaking_until
                        > before_speaking
                    ):

                        self.voiceStarted.emit()


                    # Emergency state

                    if (
                        core.emergency_until
                        > 0
                    ):

                        if (
                            core.time.time()
                            <
                            core.emergency_until
                        ):

                            self.emergencyChanged.emit(
                                True
                            )


            else:

                core.stability.update(
                    None,
                    confidence
                )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            # Clear canvas

            self.landmarksReady.emit(
                []
            )


            # Reset stability exactly
            # like original main.py.

            core.stability.update(
                None,
                0.0
            )


            self.detectionReady.emit(

                "-",

                "No hand detected",

                0.0,

                0

            )


            # Emergency timeout

            if (
                core.time.time()
                >=
                core.emergency_until
            ):

                self.emergencyChanged.emit(
                    False
                )


    # ========================================================
    # STOP
    # ========================================================

    @Slot()
    def stop(self):

        self.running = False


# ============================================================
# MAIN UI
# ============================================================

class SignSenseApp(
    QWebEngineView
):


    def __init__(self):

        super().__init__()


        # ====================================================
        # WINDOW
        # ====================================================

        self.setWindowTitle(
            "SignSense AI"
        )


        self.resize(
            1400,
            850
        )


        # ====================================================
        # SHARED FRAME
        # ====================================================

        self.shared_frame = (
            SharedFrame()
        )


        # ====================================================
        # WEB CHANNEL
        # ====================================================

        self.channel = QWebChannel(
            self.page()
        )


        self.channel.registerObject(
            "bridge",
            self
        )


        self.page().setWebChannel(
            self.channel
        )


        # ====================================================
        # HTML
        # ====================================================

        html_path = (
            ROOT /
            "ui" /
            "index.html"
        )


        self.setUrl(
            html_path.as_uri()
        )


        self.loadFinished.connect(
            self.on_loaded
        )


        # ====================================================
        # CAMERA THREAD
        # ====================================================

        self.camera_thread = (
            QThread()
        )


        self.camera_worker = (
            CameraWorker(
                self.shared_frame
            )
        )


        self.camera_worker.moveToThread(
            self.camera_thread
        )


        self.camera_thread.started.connect(
            self.camera_worker.start
        )


        self.camera_worker.frameReady.connect(
            self.update_camera
        )


        self.camera_worker.cameraReady.connect(
            self.camera_ready
        )


        self.camera_worker.error.connect(
            self.worker_error
        )


        # ====================================================
        # ML THREAD
        # ====================================================

        self.ml_thread = (
            QThread()
        )


        self.ml_worker = (
            MLWorker(
                self.shared_frame
            )
        )


        self.ml_worker.moveToThread(
            self.ml_thread
        )


        self.ml_thread.started.connect(
            self.ml_worker.start
        )


        self.ml_worker.detectionReady.connect(
            self.update_detection
        )


        self.ml_worker.landmarksReady.connect(
            self.update_landmarks
        )


        self.ml_worker.voiceStarted.connect(
            self.voice_speaking
        )


        self.ml_worker.voiceReady.connect(
            self.voice_ready
        )


        self.ml_worker.emergencyChanged.connect(
            self.show_emergency
        )


        self.ml_worker.error.connect(
            self.worker_error
        )


    # ========================================================
    # HTML LOADED
    # ========================================================

    def on_loaded(
        self,
        success
    ):

        if not success:

            print(
                "UI could not load."
            )

            return


        print(
            "SignSense AI UI loaded."
        )


        # Start camera

        self.camera_thread.start()


        # Start ML

        self.ml_thread.start()


    # ========================================================
    # CAMERA READY
    # ========================================================

    @Slot()
    def camera_ready(self):

        print(
            "Camera ready."
        )


    # ========================================================
    # CAMERA → JAVASCRIPT
    #
    # IMPORTANT:
    # This ONLY updates the camera.
    #
    # It does NOT overwrite gesture/confidence.
    # ========================================================

    @Slot(str)
    def update_camera(
        self,
        image_data
    ):

        javascript = f"""

            if (
                window.signsense &&
                window.signsense.updateCamera
            ) {{

                window.signsense.updateCamera(
                    {json.dumps(image_data)}
                );

            }}

        """


        self.page().runJavaScript(
            javascript
        )


    # ========================================================
    # MODEL → JAVASCRIPT
    # ========================================================

    @Slot(
        str,
        str,
        float,
        int
    )
    def update_detection(
        self,
        gesture,
        message,
        confidence,
        hand_count
    ):

        javascript = f"""

            if (
                window.signsense &&
                window.signsense.updateDetection
            ) {{

                window.signsense.updateDetection(

                    {json.dumps(gesture)},

                    {json.dumps(message)},

                    {confidence},

                    {hand_count}

                );

            }}

        """


        self.page().runJavaScript(
            javascript
        )


    # ========================================================
    # LANDMARKS → JAVASCRIPT
    # ========================================================

    @Slot(list)
    def update_landmarks(
        self,
        hands
    ):

        javascript = f"""

            if (
                window.signsense &&
                window.signsense.updateLandmarks
            ) {{

                window.signsense.updateLandmarks(

                    {json.dumps(hands)}

                );

            }}

        """


        self.page().runJavaScript(
            javascript
        )


    # ========================================================
    # VOICE SPEAKING
    # ========================================================

    @Slot()
    def voice_speaking(self):

        self.page().runJavaScript(

            """

            if (
                window.signsense &&
                window.signsense.voiceSpeaking
            ) {

                window.signsense.voiceSpeaking();

            }

            """

        )


    # ========================================================
    # VOICE READY
    # ========================================================

    @Slot()
    def voice_ready(self):

        self.page().runJavaScript(

            """

            if (
                window.signsense &&
                window.signsense.voiceReady
            ) {

                window.signsense.voiceReady();

            }

            """

        )


    # ========================================================
    # EMERGENCY
    # ========================================================

    @Slot(bool)
    def show_emergency(
        self,
        show
    ):

        self.page().runJavaScript(

            f"""

            if (
                window.signsense &&
                window.signsense.showEmergency
            ) {{

                window.signsense.showEmergency(
                    {str(show).lower()}
                );

            }}

            """

        )


    # ========================================================
    # ERROR
    # ========================================================

    @Slot(str)
    def worker_error(
        self,
        message
    ):

        print(
            "Worker error:",
            message
        )


    # ========================================================
    # CLOSE
    # ========================================================

    def closeEvent(
        self,
        event
    ):

        print(
            "Stopping SignSense AI..."
        )


        try:

            # Stop workers

            self.camera_worker.stop()

            self.ml_worker.stop()


            # Stop threads

            self.camera_thread.quit()

            self.ml_thread.quit()


            # Wait

            self.camera_thread.wait(
                3000
            )

            self.ml_thread.wait(
                3000
            )


        except Exception as e:

            print(
                "Shutdown error:",
                e
            )


        event.accept()


# ============================================================
# START APPLICATION
# ============================================================

app = QApplication(
    sys.argv
)


window = SignSenseApp()


window.show()


sys.exit(
    app.exec()
)