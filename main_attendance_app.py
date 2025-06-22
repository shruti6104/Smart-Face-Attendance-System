import cv2
import numpy as np
import pyttsx3
import os
from datetime import datetime

# === INIT SETUP ===
engine = pyttsx3.init()

# Load recognizer and cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load Names
names = {}
if os.path.exists("names.txt"):
    with open("names.txt", "r") as f:
        for line in f:
            id, name = line.strip().split(",")
            names[int(id)] = name

# Mark attendance in CSV
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
            print(f"[✅] Marked attendance for {name} at {now}")
            engine.say(f"Welcome {name}")
            engine.runAndWait()

# Recognition logic
def recognize_faces(frame, gray):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 60:
            name = names.get(id, "Unknown")
            label = f"{name} ({round(100 - confidence)}%)"
            mark_attendance(name)
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

# === Choose Mode ===
print("\n📌 Choose Mode:")
print("1. Live Webcam Recognition")
print("2. Photo Upload Recognition")
choice = input("Enter 1 or 2: ")

# === MODE 1: WEBCAM ===
if choice == "1":
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # safer on Windows
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to access webcam")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed = recognize_faces(frame, gray)

        cv2.imshow("🎦 Webcam Attendance", processed)
        if cv2.waitKey(1) == 27:  # ESC key
            break
    cap.release()
    cv2.destroyAllWindows()

# === MODE 2: PHOTO UPLOAD ===
elif choice == "2":
    print("\n📁 Available images in 'uploads/' folder:")
    for img in os.listdir("uploads"):
        print(" -", img)
    
    img_name = input("Enter image file name from above (e.g., photo.jpg): ")
    img_path = os.path.join("uploads", img_name)

    if not os.path.exists(img_path):
        print(f"[❌] File not found in uploads/: {img_name}")
        exit()

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = recognize_faces(img, gray)

    cv2.imshow("📸 Image Recognition", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print("❌ Invalid input. Please enter 1 or 2.")
