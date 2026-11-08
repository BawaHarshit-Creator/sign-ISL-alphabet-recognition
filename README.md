# ISL Alphabet Recognition

A Python-based project that recognizes Indian Sign Language (ISL) alphabet gestures using a webcam.

I built this project to explore how computer vision and machine learning can work together to recognize hand gestures in real time.

The system detects hand landmarks using MediaPipe and passes the extracted landmark coordinates to a Random Forest model, which predicts the corresponding alphabet.

## What it does

- Detects hands through a webcam.
- Extracts hand landmarks using MediaPipe.
- Collects and stores gesture data for training.
- Recognizes ISL alphabets from A to Z.
- Displays predictions in real time.

## Built With

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy
- Pandas

## Model Details

| Detail | Value |
|---|---|
| Model | Random Forest |
| Dataset size | 1,040 samples |
| Classes | 26 alphabets |
| Features | 126 |
| Test accuracy | 95.19% |

## Running the Project

First, clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Navigate into the project folder:

```bash
cd sign-ISL-alphabet-recognition
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the live recognition program:

```bash
python live_prediction.py
```

Make sure your webcam is connected.

## Project Structure

- `hand_detection.py` – Hand detection and landmark extraction.
- `collect_data.py` – Collects alphabet gesture samples.
- `check_dataset.py` – Checks the collected dataset.
- `train_model.py` – Trains the classification model.
- `live_prediction.py` – Performs real-time alphabet recognition.
- `webcam_test.py` – Tests webcam functionality.

## Current Limitations

This project currently focuses on alphabet recognition. It does not translate complete sentences or recognize the full vocabulary of Indian Sign Language.

## Future Scope

- Recognition of dynamic gestures.
- Word and sentence formation.
- Testing with different users and lighting conditions.

## Author

**Harshit Bawa**