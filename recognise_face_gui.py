import cv2
import numpy as np
import pyttsx3
from datetime import datetime
import os

# Load trained model and face detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load names from names.txt
names = {}
if os.path.exists("names.txt"):
    with open("names.txt", "r") as f:
        for line in f.readlines():
            id, name = line.strip().split(",")
            names[int(id)] = name

# Voice init
engine = pyttsx3.init()

# Attendance setup
def mark_attendance(name):
    if not os.path.exists("attendance.csv"):
        with open("attendance.csv", "w") as f:
            f.write("Name,Time\n")

    with open("attendance.csv", "r+") as f:
        lines = f.readlines()
        logged = [line.split(",")[0] for line in lines]
        if name not in logged:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{name},{now}\n")
            engine.say(f"Welcome {name}")
            engine.runAndWait()

# Webcam
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        face_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 60:
            name = names.get(face_id, "Unknown")
            label = f"{name} ({round(100 - confidence)}%)"
            mark_attendance(name)
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("🎦 Face Recognition Attendance", frame)
    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cam.release()
cv2.destroyAllWindows()
