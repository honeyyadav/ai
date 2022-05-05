import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time
from math import dist

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Curl counter variables
counter = 0
stage = None
secondsPassed = 0
currentTime = time.time()

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("assests/rope_swing.mp4")

xAxisPlotLeft = []
yAxisPlotLeft = []
xAxisPlotRight = []
yAxisPlotRight = []


# naming the x axis
plt.xlabel("Time(s)")
# naming the y axis
plt.ylabel("Height")

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not (ret):
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            video_width = cap.get(3)
            video_height = cap.get(4)

            left_foot = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_FOOT_INDEX
            ]
            right_foot = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
            ]
            left_thumb = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.LEFT_THUMB
            ]
            right_thumb = results.pose_landmarks.landmark[
                mp_pose.PoseLandmark.RIGHT_THUMB
            ]

            LEFT_FOOT_INDEX_custom_x = round(left_foot.x * video_width)
            LEFT_FOOT_INDEX_custom_y = round(left_foot.y * video_height)
            RIGHT_FOOT_INDEX_custom_x = round(right_foot.x * video_width)
            RIGHT_FOOT_INDEX_custom_y = round(right_foot.y * video_height)

            LEFT_THUMB_custom_x = round(left_thumb.x * video_width)
            LEFT_THUMB_custom_y = round(left_thumb.y * video_height)
            RIGHT_THUMB_custom_x = round(right_thumb.x * video_width)
            RIGHT_THUMB_custom_y = round(right_thumb.y * video_height)

            leftHeight = left_thumb.y * 1000

            xAxisPlotLeft.append(time.time() - currentTime)
            yAxisPlotLeft.append(left_thumb.y * 1000)
            xAxisPlotRight.append(time.time() - currentTime)
            yAxisPlotRight.append(right_thumb.y * 1000)

            # Curl counter logic
            if leftHeight > 280:
                stage = "up"
            if leftHeight < 240 and stage == "up":
                stage = "down"
                counter += 1

            cv2.line(
                image,
                (LEFT_FOOT_INDEX_custom_x, LEFT_FOOT_INDEX_custom_y),
                (LEFT_THUMB_custom_x, LEFT_THUMB_custom_y),
                (0, 255, 0),
                thickness=2,
            )

            cv2.line(
                image,
                (RIGHT_FOOT_INDEX_custom_x, RIGHT_FOOT_INDEX_custom_y),
                (RIGHT_THUMB_custom_x, RIGHT_THUMB_custom_y),
                (0, 0, 255),
                thickness=2,
            )

        except:
            pass

        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0, 0), (125, 73), (245, 117, 16), -1)

        # Rep data
        cv2.putText(
            image,
            "COUNT",
            (15, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            (str(counter)).rjust(2, "0"),
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if ret:
            cv2.imshow("Mediapipe Feed", image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # plotting the points
    plt.plot(
        xAxisPlotLeft,
        yAxisPlotLeft,
        color="green",
        linestyle="dashed",
        linewidth=1,
        label="Left Hand",
    )

    # plotting the points
    plt.plot(
        xAxisPlotRight,
        yAxisPlotRight,
        color="red",
        linestyle="dashed",
        linewidth=1,
        label="Right Right",
    )

    # giving a title to my graph
    plt.title("Excercise done - " + str(counter) + " Times in " + str(round(time.time() - currentTime)) + " Seconds")
    plt.legend()
    # function to show the plot
    plt.show()
