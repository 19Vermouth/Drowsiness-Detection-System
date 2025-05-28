import cv2
import dlib
import time
import threading
from scipy.spatial import distance
import pygame
import os

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")
ALERT_SOUND = os.path.join(BASE_DIR, "alert_40hz.wav")
VIDEO_PATH = os.path.join(BASE_DIR, "driver_video.mp4")

# Thresholds and frame count
EAR_THRESHOLD = 0.25
YAWN_THRESHOLD = 20
CONSEC_FRAMES = 20

# Init dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Eye and mouth landmarks
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)
(mStart, mEnd) = (60, 68)

# Sound setup
pygame.mixer.init()
def sound_alert():
    pygame.mixer.music.load(ALERT_SOUND)
    pygame.mixer.music.play()

# EAR and MAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_opening(mouth):
    A = distance.euclidean(mouth[3], mouth[7])  # 63, 67
    B = distance.euclidean(mouth[5], mouth[1])  # 65, 61
    mar = (A + B) / 2.0
    return mar

# Counters
frame_counter = 0
alarm_on = False

# Start video
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mar = mouth_opening(mouth)

        for (x, y) in leftEye + rightEye + mouth:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSEC_FRAMES and not alarm_on:
                alarm_on = True
                threading.Thread(target=sound_alert, daemon=True).start()
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            frame_counter = 0
            alarm_on = False

        if mar > YAWN_THRESHOLD:
            if not alarm_on:
                alarm_on = True
                threading.Thread(target=sound_alert, daemon=True).start()
            cv2.putText(frame, "YAWN ALERT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
