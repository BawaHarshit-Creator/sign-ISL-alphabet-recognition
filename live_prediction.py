import cv2
import mediapipe as mp
import time
import pickle
import numpy as np

from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# 1. LOAD TRAINED MODEL

with open("isl_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("label_encoder.pkl", "rb") as file:
    encoder = pickle.load(file)


# 2. LOAD MEDIAPIPE

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)


# 3. HAND CONNECTIONS

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

ZERO_HAND = [0.0] * 63


# 4. START WEBCAM

cap = cv2.VideoCapture(0)

start_time = time.perf_counter()

predictions = deque(maxlen=10)

print("ISL Live Recognition Started!")
print("Press Q to quit.")


# 5. MAIN LOOP

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    height, width, _ = frame.shape

    hand_data = {
        "Left": None,
        "Right": None
    }


    # 6. PROCESS DETECTED HANDS

    if result.hand_landmarks:

        for index, hand in enumerate(result.hand_landmarks):

            hand_label = (
                result.handedness[index][0].category_name
            )

            # Correct labels for mirrored camera
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

            # Draw hand connections
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

            cv2.putText(
                frame,
                hand_label,
                (int(hand[0].x * width), int(hand[0].y * height) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


    # 7. PREDICT ALPHABET

    if result.hand_landmarks:

        left_data = hand_data["Left"] or ZERO_HAND
        right_data = hand_data["Right"] or ZERO_HAND

        features = np.array([
            left_data + right_data
        ])

        probabilities = model.predict_proba(features)[0]

        best_index = np.argmax(probabilities)

        confidence = probabilities[best_index]

        predicted_number = model.classes_[best_index]

        predicted_letter = encoder.inverse_transform(
            [predicted_number]
        )[0]

        if confidence >= 0.60:

            predictions.append(predicted_letter)

            stable_prediction = max(
                set(predictions),
                key=predictions.count
            )

            display_text = (
                f"Prediction: {stable_prediction} "
                f"({confidence * 100:.1f}%)"
            )

        else:
            predictions.clear()
            display_text = "Uncertain"

        cv2.putText(
            frame,
            display_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    else:
        predictions.clear()

        cv2.putText(
            frame,
            "Show an ISL gesture",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # 8. DISPLAY

    cv2.imshow("ISL Live Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# 9. RELEASE RESOURCES

cap.release()
detector.close()
cv2.destroyAllWindows()

print("Recognition stopped.")