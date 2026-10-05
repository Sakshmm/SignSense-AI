# SignSense AI

A beginner-friendly AI/ML + Computer Vision project for real-time custom hand-gesture recognition using a webcam.

You collect your own gesture data, train your own classifier, and get real-time gesture → text → voice output.

> Note: This project does NOT use any pretrained sign-language classification model. The dataset and the model are entirely yours.

---

## Pipeline

```
Webcam → MediaPipe Hand Landmarks → Normalized Features → Custom Random Forest → Gesture → Text → Voice
```

---

## 1. Requirements

Recommended for this project:

- Python 3.10 or 3.11
- MediaPipe 0.10.21 (intentionally pinned — see Why MediaPipe is pinned to 0.10.21)
- A working webcam
- Windows / macOS / Linux
- Internet only for installing Python packages

### Full dependency list

See `requirements.txt`. It includes:

- `mediapipe==0.10.21`
- `opencv-python`
- `numpy`
- `scikit-learn`
- `pyttsx3`

Linux users: `pyttsx3` may require `espeak` or `espeak-ng`. Install with:

```bash
sudo apt install espeak-ng
```

---

## 2. Create a virtual environment

### Windows — PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Windows — CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Make sure your virtual environment is activated before running this.

---

## 4. Dataset collection

Run:

```bash
python collect_data.py
```

You will be prompted to type one label at a time. Examples:

```text
HELLO
OK
STOP
YES
NO
HELP
WATER
FOOD
PAIN
SOS
A
B
C
...
Z
```

### How the collector works

1. Type a label (e.g. `HELLO`) and press Enter.
2. The webcam window opens.
3. Press SPACE to start/pause recording.
4. Move your hand slightly while recording (see advice below).
5. Press ESC to stop the current label.
6. Type the next label and repeat.

Samples are saved as:

```text
dataset/
├── HELLO.csv
├── OK.csv
├── ...
└── Z.csv
```

Default is 300 samples per gesture. Change `SAMPLES_PER_GESTURE` in `config/settings.json` if needed.

### Dataset advice

For each gesture, move your hand slightly and collect samples with:

- Different positions in the frame
- Slightly different distances from the camera
- Normal lighting variations
- Natural hand orientation

Do not collect fake or random data. These CSV files are your real custom dataset — the model's accuracy depends entirely on them.

---

## 5. Train your model

After collecting at least two classes:

```bash
python train_model.py
```

The script will:

1. Load all CSV files from `dataset/`
2. Combine them into one training set
3. Split into train/test
4. Train a Random Forest classifier
5. Print accuracy
6. Print a classification report
7. Save the trained model and label encoder

### Output

```text
model/
├── gesture_model.pkl
├── label_encoder.pkl
└── model_info.json
```

`model_info.json` contains metadata such as:

```json
{
  "trained_at": "2025-01-15T10:30:00",
  "classes": ["HELLO", "OK", "STOP"],
  "accuracy": 0.964,
  "n_samples": 2700
}
```

You can retrain any time — running `train_model.py` replaces the old model files.

---

## 6. Run the real-time application

```bash
python main.py
```

### Controls

#### Switching modes

| Key   | Action        |
|-------|---------------|
| `W`   | Word Mode     |
| `S`   | Spelling Mode |
| `ESC` | Exit the app  |

#### Word Mode

Recognizes predefined gestures (like `HELP`, `WATER`, `SOS`) and speaks a mapped sentence.

Example:

```
Gesture: HELP  →  Spoken/Displayed: "I need a nurse"
```

The mapping is defined in `config/gestures.json` (see Section 8).

#### Spelling Mode

Recognizes individual alphabet letters and builds a word/sentence one letter at a time.

| Key         | Action                |
|-------------|-----------------------|
| `SPACE`     | Add a space           |
| `BACKSPACE` | Delete last character |
| `C`         | Clear current text    |
| `ENTER`     | Speak the current text |

Example:

```
G → E → E → T → I → K → A   becomes   "GEETIKA"
```

To enter the same letter twice (e.g. `LL` in `HELLO`), do not hold the same gesture continuously. Show the letter, move/change/remove your hand, then show the same letter again.

---

## 7. Configurable settings

Edit:

```text
config/settings.json
```

Default values:

```json
{
    "CONFIDENCE_THRESHOLD": 0.75,
    "STABLE_FRAMES": 5,
    "COOLDOWN_SECONDS": 2.0,
    "CAMERA_INDEX": 0,
    "MAX_HANDS": 1,
    "SAMPLES_PER_GESTURE": 300
}
```

### Meaning

| Key                    | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `CONFIDENCE_THRESHOLD` | Minimum ML probability required to accept a prediction       |
| `STABLE_FRAMES`        | Consecutive frames required before accepting a prediction    |
| `COOLDOWN_SECONDS`     | Prevents rapid repeated triggers                             |
| `CAMERA_INDEX`         | `0` = default webcam; use `1` if another camera is selected  |
| `MAX_HANDS`            | Maximum number of hands tracked simultaneously               |
| `SAMPLES_PER_GESTURE`  | Number of samples collected for each label during collection |

---

## 8. Changing word messages

Edit:

```text
config/gestures.json
```

Example:

```json
{
    "HELP": "I need a nurse",
    "WATER": "Please give me water",
    "SOS": "Emergency! Please help me"
}
```

The model label stays the same (e.g. `HELP`), but the spoken/displayed message changes. This lets you customize output without retraining.

---

## 9. Adding or removing alphabet labels

Edit:

```text
config/alphabet.json
```

This file defines which letters are valid in Spelling Mode. The collector accepts the label you type, so you can train only the classes you currently need.

Example — if you only want to train 5 letters:

```json
["A", "B", "C", "D", "E"]
```

The classifier will only ever predict these letters.

---

## 10. Important training rule

The exact same feature extraction function is used in:

```text
utils/feature_extraction.py
```

It is imported by both:

```text
collect_data.py
main.py
```

This is critical — training and prediction must use the same feature format. If you change the feature extractor, you must recollect the dataset and retrain the model.

---

## 11. Troubleshooting

### Camera does not open

1. Open `config/settings.json` and change:

```json
"CAMERA_INDEX": 1
```

2. Close any other apps using the webcam (Zoom, Teams, browser tabs, etc.).
3. On macOS, grant camera permission to Terminal / VS Code in System Settings → Privacy → Camera.

### Model not found

Run the full pipeline:

```bash
python collect_data.py
python train_model.py
python main.py
```

### Accuracy is low

- Collect more real samples per class (aim for 300+).
- Make gestures visually distinct from each other.
- Avoid collecting all classes in exactly one hand position.
- Check that lighting is not too dim or too harsh.

### Voice does not work

- `pyttsx3` uses your local system speech engine.
- Windows: usually works out of the box (SAPI5).
- macOS: uses the `say` command — usually works.
- Linux: may need `sudo apt install espeak-ng`.

### MediaPipe import error

Ensure you installed the pinned version:

```bash
pip install mediapipe==0.10.21
```

---

## 12. Project structure

```text
SignSense-AI/
│
├── main.py
├── collect_data.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── config/
│   ├── gestures.json
│   ├── alphabet.json
│   └── settings.json
│
├── dataset/
│   └── .gitkeep
│
├── model/
│   └── .gitkeep
│
├── utils/
│   ├── __init__.py
│   ├── hand_tracking.py
│   ├── feature_extraction.py
│   ├── gesture_stability.py
│   ├── speech.py
│   └── history.py
│
└── screenshots/
    └── .gitkeep
```

---

## 13. What is intentionally NOT included

- React / Next.js
- Node.js
- FastAPI / Flask
- Database
- Docker
- Cloud APIs (AWS, GCP, Azure)
- Email / SMS services
- Any pretrained sign-language classifier
- Fake or synthetic training data

The focus is intentionally on:

> AI/ML + Computer Vision + Custom Dataset + Local Speech

---

## 14. Why MediaPipe is pinned to 0.10.21

This project intentionally uses the beginner-friendly `mp.solutions.hands` API.

MediaPipe removed the legacy `mp.solutions` Python API from version 0.10.31 onward. Using the latest MediaPipe release would require:

- Rewriting the hand-tracking code with the more complex Tasks API
- Downloading additional `.task` model assets
- More boilerplate for beginners

Version 0.10.21 keeps this project simple, matches the requested architecture, and works out of the box.

---

## License

This project is for educational purposes. Add your own license if you plan to distribute it.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.