import pandas as pd
from pathlib import Path

dataset_path = Path(__file__).resolve().parent / "isl_landmarks_new.csv"
df = pd.read_csv(dataset_path)

print("Dataset file:", dataset_path)
print("Dataset shape:", df.shape)

print("\nSamples per alphabet:")
print(df["label"].value_counts().sort_index())

print("\nMissing values:", df.isnull().sum().sum())

print("\nFirst 5 rows:")
print(df.head())