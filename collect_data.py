import cv2
import mediapipe as mp
import time
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# CONFIGURATION
DATASET_FILE = "isl_landmarks_new.csv"
SAMPLES_PER_LETTER = 40
CAPTURE_INTERVAL = 1.5

HAND_FEATURES = 63
ZERO_HAND = [0.0] * HAND_FEATURES

# LOAD MEDIAPIPE
base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


# HAND CONNECTIONS
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# CSV HEADERS
header = ["label"]

for hand_name in ["left", "right"]:
    for i in range(21):
        header.extend([
            f"{hand_name}_x{i}",
            f"{hand_name}_y{i}",
            f"{hand_name}_z{i}"
        ])


if not os.path.exists(DATASET_FILE):
    with open(DATASET_FILE, "w", newline="") as file:
        csv.writer(file).writerow(header)


# WEBCAM
cap = cv2.VideoCapture(0)
start_time = time.perf_counter()

current_letter = None
sample_count = 0
next_capture_time = 0

print("ISL DATA COLLECTION")
print("Press A-Z to select an alphabet.")
print("40 samples will be captured automatically.")
print("Press ESC to exit.")


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame, cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    result = detector.detect_for_video(
        mp_image, timestamp_ms
    )

    height, width, _ = frame.shape

    hand_data = {
        "Left": None,
        "Right": None
    }


    # PROCESS HANDS
    if result.hand_landmarks:

        for index, hand in enumerate(result.hand_landmarks):

            hand_label = (
                result.handedness[index][0].category_name
            )

            # Correct mirrored handedness
            if hand_label == "Left":
                hand_label = "Right"
            else:
                hand_label = "Left"

            landmarks = []

            for landmark in hand:
                landmarks.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            hand_data[hand_label] = landmarks

            # Draw connections
            for start, end in HAND_CONNECTIONS:

                x1 = int(hand[start].x * width)
                y1 = int(hand[start].y * height)

                x2 = int(hand[end].x * width)
                y2 = int(hand[end].y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

            # Draw landmarks
            for landmark in hand:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


    # AUTOMATIC COLLECTION
    if current_letter is not None:

        if sample_count < SAMPLES_PER_LETTER:

            if (
                result.hand_landmarks
                and time.perf_counter() >= next_capture_time
            ):

                left_data = (
                    hand_data["Left"] or ZERO_HAND
                )

                right_data = (
                    hand_data["Right"] or ZERO_HAND
                )

                row = [
                    current_letter,
                    *left_data,
                    *right_data
                ]

                with open(
                    DATASET_FILE, "a", newline=""
                ) as file:

                    csv.writer(file).writerow(row)

                sample_count += 1

                next_capture_time = (
                    time.perf_counter() + CAPTURE_INTERVAL
                )

                print(
                    f"{current_letter}: "
                    f"{sample_count}/{SAMPLES_PER_LETTER}"
                )

        else:
            print(
                f"{current_letter} completed! "
                "Press the next alphabet."
            )

            current_letter = None


    # DISPLAY STATUS
    if current_letter is None:

        display_text = "Press A-Z to start"

    else:

        display_text = (
            f"Letter: {current_letter} | "
            f"Samples: {sample_count}/40"
        )

    cv2.putText(
        frame,
        display_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "ESC = Exit",
        (20, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("ISL Dataset Collection", frame)


    # KEYBOARD INPUT
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if (
        current_letter is None
        and (65 <= key <= 90 or 97 <= key <= 122)
    ):

        current_letter = chr(key).upper()
        sample_count = 0
        next_capture_time = time.perf_counter()

        print(f"\nStarted collecting {current_letter}")


# CLEANUP
cap.release()
detector.close()
cv2.destroyAllWindows()

print("\nDataset collection finished!")