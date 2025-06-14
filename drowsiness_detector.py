
import cv2
import mediapipe as mp
import time
import threading
from scipy.spatial import distance
import pygame
import os

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALERT_SOUND = os.path.join(BASE_DIR, "alert_40hz.wav")

# Thresholds and frame count
EAR_THRESHOLD = 0.20  # Adjusted for eye closure detection
YAWN_THRESHOLD = 30   # Increased to reduce false positives (tune between 28-35)
CONSEC_FRAMES = 20
BLINK_EAR_THRESHOLD = 0.25  # EAR below this indicates a blink, suppress yawn alerts

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# MediaPipe indices for eyes and mouth
LEFT_EYE = [33, 159, 158, 133, 145, 153]  # Outer, top, top-near, inner, bottom, bottom-near
RIGHT_EYE = [263, 386, 385, 362, 374, 380]  # Outer, top, top-near, inner, bottom, bottom-near
MOUTH = [13, 78, 308, 14, 80, 310]         # Inner lips: top (13, 78, 308), bottom (14, 80, 310)

# Sound setup
pygame.mixer.init()
def sound_alert():
    pygame.mixer.music.load(ALERT_SOUND)
    pygame.mixer.music.play()

# EAR and MAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[4])  # Top center (159/386) to bottom center (145/374)
    B = distance.euclidean(eye[2], eye[5])  # Top near (158/385) to bottom near (153/380)
    C = distance.euclidean(eye[0], eye[3])  # Outer (33/263) to inner (133/362)
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_opening(mouth):
    # Use single vertical distance for stability (center top to bottom)
    A = distance.euclidean(mouth[0], mouth[3])  # 13-14 (center top to bottom)
    return A

# Counters
frame_counter = 0
alarm_on = False

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    frame = cv2.resize(frame, (640, 360))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            shape = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            leftEye = [shape[i] for i in LEFT_EYE]
            rightEye = [shape[i] for i in RIGHT_EYE]
            mouth = [shape[i] for i in MOUTH]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_opening(mouth)

            # Print EAR and MAR for calibration
            print(f"EAR: {ear:.2f}, MAR: {mar:.2f}, Blink: {ear < BLINK_EAR_THRESHOLD}")

            # Draw eye landmarks with indices
            for idx, (x, y) in zip(LEFT_EYE + RIGHT_EYE, leftEye + rightEye):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                cv2.putText(frame, str(idx), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Draw mouth landmarks with indices
            for idx, (x, y) in zip(MOUTH, mouth):
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                cv2.putText(frame, str(idx), (x + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            # Drowsiness detection
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

            # Yawn detection (suppress if blinking)
            if mar > YAWN_THRESHOLD and ear >= BLINK_EAR_THRESHOLD:
                if not alarm_on:
                    alarm_on = True
                    threading.Thread(target=sound_alert, daemon=True).start()
                cv2.putText(frame, "YAWN ALERT!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display EAR, MAR, and thresholds
            cv2.putText(frame, f"EAR: {ear:.2f} (Thr: {EAR_THRESHOLD})", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f} (Thr: {YAWN_THRESHOLD})", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
