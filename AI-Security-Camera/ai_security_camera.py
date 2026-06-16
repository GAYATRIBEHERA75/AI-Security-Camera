import cv2
import numpy as np
import datetime
import os

# Create folders if they don't exist
os.makedirs("snapshots", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Start camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Read first frame
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Video recording setup
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
video_path = f"recordings/security_{timestamp}.avi"

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(
    video_path,
    fourcc,
    20.0,
    (640, 480)
)

saved = False

while cap.isOpened():

    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    _, thresh = cv2.threshold(
        blur,
        20,
        255,
        cv2.THRESH_BINARY
    )

    dilated = cv2.dilate(
        thresh,
        None,
        iterations=3
    )

    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    for contour in contours:

        if cv2.contourArea(contour) < 1000:
            continue

        motion_detected = True

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            frame1,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    # Face Detection
    gray_frame = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray_frame,
        1.3,
        5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame1,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # Date & Time
    current_time = datetime.datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    cv2.putText(
        frame1,
        current_time,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Motion Alert
    if motion_detected:

        cv2.putText(
            frame1,
            "MOTION DETECTED!",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        # Save snapshot once
        if not saved:

            image_name = (
                f"snapshots/motion_{timestamp}.jpg"
            )

            cv2.imwrite(
                image_name,
                frame1
            )

            saved = True

    else:
        saved = False

    # Record video
    video_writer.write(frame1)

    cv2.imshow(
        "AI Security Camera",
        frame1
    )

    frame1 = frame2
    ret, frame2 = cap.read()

    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_writer.release()
cap.release()
cv2.destroyAllWindows()