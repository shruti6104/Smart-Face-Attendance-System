
import cv2
import os
import pyttsx3
from datetime import datetime

# === Initialize text-to-speech engine ===
engine = pyttsx3.init()

# === Paths ===
UPLOAD_DIR = "uploads"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
TRAINER_PATH = "trainer.yml"
NAMES_FILE = "names.txt"
ATTENDANCE_FILE = "attendance.csv"

# === Ensure uploads folder exists ===
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    print(f"[INFO] Created missing folder: {UPLOAD_DIR}")

# === Load recognizer and cascade ===
if not os.path.exists(TRAINER_PATH):
    print("[❌ ERROR] trainer.yml not found. Please train the model first.")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# === Load names from names.txt ===
names = {}
if os.path.exists(NAMES_FILE):
    with open(NAMES_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if "," in line:
                try:
                    id_str, name = line.split(",", 1)
                    names[int(id_str)] = name
                except ValueError:
                    continue
else:
    print("[❌ ERROR] names.txt file not found.")
    exit()

# === Function to mark attendance ===
def mark_attendance(name):
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            f.write("Name,Time\n")

    with open(ATTENDANCE_FILE, "r+") as f:
        lines = f.readlines()
        logged_names = [line.split(",")[0] for line in lines if "," in line]
        if name not in logged_names:
            time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{name},{time_now}\n")
            engine.say(f"Welcome {name}")
            engine.runAndWait()
            print(f"[✅ INFO] Attendance marked for: {name}")

# === Process all images in uploads/ ===
images = [img for img in os.listdir(UPLOAD_DIR) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not images:
    print(f"[⚠️ WARNING] No image files found in {UPLOAD_DIR}/")
else:
    for image_name in images:
        image_path = os.path.join(UPLOAD_DIR, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[⚠️ WARNING] Could not read image: {image_name}")
            continue

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

