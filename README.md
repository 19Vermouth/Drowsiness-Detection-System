# Drowsiness Detection System 😴🚗

## 🔍 Overview

The **Drowsiness Detection System** was created to reduce the number of road accidents caused by driver fatigue and inattention. 
This intelligent system uses computer vision and machine learning techniques to monitor the driver's eye state in real-time and trigger an alert if drowsiness is detected.

---

## 🎯 Purpose

Driver fatigue is a leading cause of vehicle accidents worldwide.
This project provides a low-cost, effective solution using a standard webcam and Python to detect early signs of drowsiness and provide timely alerts, ensuring driver safety.

---

## 🚀 Features

- Real-time video stream monitoring.
- Eye state detection using Haar Cascades and deep learning.
- Alarm system that triggers when drowsiness is detected.
- Lightweight and runs on standard hardware.

---

## 🧠 Technologies 
| ⚙️ Technology                    | 📘 Description                                                              |
| -------------------------------- | --------------------------------------------------------------------------- |
| **Python**                       | Core programming language used for the entire application                   |
| **OpenCV**                       | Used for webcam feed capture and drawing visual overlays on the video frame |
| **MediaPipe (Face Mesh)**        | ML-based solution to track facial landmarks in real time                    |
| **EAR & MAR Algorithms**         | Eye Aspect Ratio and Mouth Aspect Ratio to determine drowsiness or yawning  |
| **Threading**                    | Enables non-blocking alert sound playback during detection                  |
| **Virtual Environment (`venv`)** | Isolates dependencies for clean and reproducible development                |


---

##  📦Libraries Used
| 📦 Library               | 🔍 Purpose                                                                          |
| ------------------------ | ----------------------------------------------------------------------------------- |
| `cv2` (OpenCV)           | Real-time video capture, frame processing, and drawing facial annotations           |
| `mediapipe`              | Facial landmark detection (eyes, mouth, etc.) using the Face Mesh model             |
| `pygame`                 | Plays audio alerts (alarm when drowsiness or yawning is detected)                   |
| `scipy.spatial.distance` | Computes Euclidean distance for EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) |
| `threading`              | Allows sound alert to run in a separate thread without blocking the video feed      |
| `time`                   | For optional delays or future logging/timestamp features                            |
| `os`                     | Helps in creating platform-independent paths to the alert sound file                |


---

## 💻 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/19Vermouth/Drowsiness-Detection-System.git
cd Drowsiness-Detection-System
```
### 2.  Create a Virtual Environment
python -m venv venv

### 3. Activate the Environment
venv\Scripts\activate

### 4. Install Required Packages
Make sure requirements.txt exists. If not, create it manually with the libraries you use. Example:
pip install -r requirements.txt

Sample requirements.txt:
opencv-python
numpy
pygame
imutils

### A.How to Run
1. Activate the virtual environment:
venv\Scripts\activate  # Windows
### OR
source venv/bin/activate  # macOS/Linux

2. Execute the main program:
   python main.py
   (Replace main.py with your actual entry file if named differently.)

## 📁 Project Structure

| Path                        | Description                                 |
|-----------------------------|---------------------------------------------|
| `venv/`                     | Virtual environment (excluded from Git)     |
| `main.py`                   | Main script for drowsiness detection        |
| `alarm.wav`                 | Alert sound file                            |
| `requirements.txt`          | List of required Python packages            |
| `README.md`                 | Project documentation                       |
| `.gitignore`                | To ignore `venv/` and unnecessary files     |

## 🙌 Contribution

If you’d like to contribute, feel free to fork the repo and submit a pull request.

---

## 📄 License

This project is open-source under the **MIT License**.

