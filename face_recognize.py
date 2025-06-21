import cv2
import os
import pyttsx3
from datetime import datetime

# Initialize TTS engine
engine = pyttsx3.init()

# Paths
path = "uploads"
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
trainer_path = "trainer.yml"
names_file = "names.txt"
attendance_file = "attendance.csv"

# Ensure uploads folder exists
if not os.path.exists(path):
    os.makedirs(path)
    print(f"[INFO] Created missing folder: {path}")

# Load face recognizer and cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load names from file
names = {}
if os.path.exists(names_file):
    with open(names_file, "r") as f:
        for line in f:
            id, name = line.strip().split(",")
            names[int(id)] = name
else:
    print("[ERROR] names.txt file not found.")
    exit()

# Function to mark attendance
def mark_attendance(name):
    if not os.path.exists(attendance_file):
        with open(attendance_file, "w") as f:
            f.write("Name,Time\n")

    with open(attendance_file, "r+") as f:
        lines = f.readlines()
        logged_names = [line.split(",")[0] for line in lines]
        if name not in logged_names:
            time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{name},{time_now}\n")
            engine.say(f"Welcome {name}")
            engine.runAndWait()
            print(f"[INFO] Attendance marked for: {name}")

# Process each image in uploads/
images = [img for img in os.listdir(path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not images:
    print(f"[WARNING] No image files found in {path}/")
else:
    for image_name in images:
        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

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

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow(f"📸 {image_name}", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

