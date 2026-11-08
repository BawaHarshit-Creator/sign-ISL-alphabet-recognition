from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "isl_landmarks_new.csv"
MODEL_PATH = PROJECT_DIR / "isl_model.pkl"
ENCODER_PATH = PROJECT_DIR / "label_encoder.pkl"

FEATURE_COLUMNS = [
    f"{hand}_{axis}{landmark}"
    for hand in ("left", "right")
    for landmark in range(21)
    for axis in ("x", "y", "z")
]


if not DATASET_PATH.is_file():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}. "
        "Check that isl_landmarks_new.csv is in the project folder."
    )

df = pd.read_csv(DATASET_PATH)
expected_columns = ["label", *FEATURE_COLUMNS]

if list(df.columns) != expected_columns:
    raise ValueError(
        "Dataset columns do not match the expected format: "
        "label followed by 126 left/right hand landmark features."
    )

if df.empty:
    raise ValueError(f"Dataset is empty: {DATASET_PATH}")

labels = df["label"].astype("string").str.strip()
if labels.isna().any() or labels.eq("").any():
    raise ValueError("Dataset contains missing or empty labels.")

class_counts = labels.value_counts()
if len(class_counts) < 2:
    raise ValueError("At least two different labels are required for training.")
if class_counts.min() < 2:
    raise ValueError(
        "Each label needs at least two samples for a stratified train/test split."
    )

X = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
if not np.isfinite(X.to_numpy()).all():
    raise ValueError(
        "Dataset contains missing, non-numeric, or infinite landmark values."
    )

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Dataset: {DATASET_PATH}")
print(f"Samples: {len(df)} | Features: {X.shape[1]} | Labels: {len(encoder.classes_)}")
print(f"Model accuracy: {accuracy * 100:.2f}%")
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
    )
)

with MODEL_PATH.open("wb") as file:
    pickle.dump(model, file)

with ENCODER_PATH.open("wb") as file:
    pickle.dump(encoder, file)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Label encoder saved to: {ENCODER_PATH}")
