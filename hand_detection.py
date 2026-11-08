import cv2
import mediapipe as mp
import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision



# 1. LOAD HAND DETECTION MODEL


base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)



# 2. INITIALIZE HAND DETECTOR


detector = vision.HandLandmarker.create_from_options(options)



# 3. HAND LANDMARK CONNECTIONS


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]



# 4. START WEBCAM


cap = cv2.VideoCapture(0)

start_time = time.perf_counter()

printed_landmarks = False



# 5. MAIN LOOP


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Cannot access webcam")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert into MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Generate timestamp
    timestamp_ms = int(
        (time.perf_counter() - start_time) * 1000
    )

    # Detect hands
    result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )

    # Get frame dimensions
    height, width, _ = frame.shape


    
    # 6. HAND DETECTION
    

    if result.hand_landmarks:

        for index, hand in enumerate(result.hand_landmarks):

            # Identify hand
            hand_label = result.handedness[index][0].category_name

            # Swap labels for mirrored display
            if hand_label == "Left":
                hand_label = "Right"
            else:
                hand_label = "Left"


            
            # 7. DISPLAY HAND LABEL
            

            wrist_x = int(hand[0].x * width)
            wrist_y = int(hand[0].y * height)

            cv2.putText(
                frame,
                hand_label,
                (wrist_x, wrist_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


            
            # 8. DRAW HAND CONNECTIONS
           

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


            
            # 9. EXTRACT LANDMARK COORDINATES
           

            landmark_data = []

            for landmark in hand:

                landmark_data.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                # Draw landmarks
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )


            
            # 10. PRINT COORDINATES
            

            if not printed_landmarks:

                print("\nExtracted Hand Landmarks:")
                print(landmark_data)

                print(
                    "Total coordinates:",
                    len(landmark_data)
                )

                printed_landmarks = True


        # Display detection status
        cv2.putText(
            frame,
            "Hand Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


    
    # 11. DISPLAY WEBCAM
    

    cv2.imshow(
        "ISL Sign Language Recognition",
        frame
    )


    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# 12. RELEASE RESOURCES


cap.release()

detector.close()

cv2.destroyAllWindows()