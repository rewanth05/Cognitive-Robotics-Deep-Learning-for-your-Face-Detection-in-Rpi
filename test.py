# # replace with your phone IP
import cv2
import os
import numpy as np

# -----------------------------
# Setup
# -----------------------------
URL = "http://10.14.25.228:8080/video"  # Replace with your phone IP
cap = cv2.VideoCapture(URL, cv2.CAP_FFMPEG)

# Use Haar cascade from local file (download if needed)
cascade_path = "./haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    import urllib.request
    print("Downloading Haar cascade...")
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, cascade_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

dataset_dir = "auto_face_dataset"
os.makedirs(dataset_dir, exist_ok=True)
model_file = "auto_face_model.yml"

# -----------------------------
# Step 1: Capture face images
# -----------------------------
print("🔹 Capturing your face. Look at the camera...")
count = 0
while count < 30:  # capture 30 frames
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(dataset_dir, f"face_{count}.jpg"), face_img)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(frame, f"Capturing... {count}/30", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Auto Face Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("✅ Captured 30 images of your face!")
cv2.destroyAllWindows()

# -----------------------------
# Step 2: Train LBPH recognizer
# -----------------------------
print("🔹 Training LBPH face recognizer...")
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

for file in os.listdir(dataset_dir):
    img = cv2.imread(os.path.join(dataset_dir, file), cv2.IMREAD_GRAYSCALE)
    faces.append(img)
    labels.append(0)  # label 0 = your face

recognizer.train(faces, np.array(labels))
recognizer.save(model_file)
print(f"✅ Model trained and saved as {model_file}")

# -----------------------------
# Step 3: Real-time recognition
# -----------------------------
print("🔹 Starting real-time face recognition...")
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_img)
        text = f"Your Face ({confidence:.0f})"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Auto Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
