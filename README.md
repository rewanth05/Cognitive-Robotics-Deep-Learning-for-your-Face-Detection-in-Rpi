# Cognitive-Robotics-Deep-Learning-for-your-Face-Detection-in-Rpi
Auto Face Detection & Recognition 📸🤖

This project lets you capture your face using your phone camera (IP webcam), train an LBPH face recognizer, and perform real-time recognition.

If the model recognizes your face, it will show a green box with confidence score.

🚀 Features

📷 Capture 30 face samples automatically.

🧠 Train an LBPH (Local Binary Patterns Histogram) face recognizer.

🔍 Perform real-time face detection & recognition.

✅ Works with Raspberry Pi 5 / PC / Laptop.

📦 Requirements

Python 3.8+

OpenCV with contrib package (opencv-contrib-python)

NumPy

Install dependencies:

pip install opencv-python opencv-contrib-python numpy

⚡ Setup
1️⃣ Connect Your Phone as Camera

Install IP Webcam (Android) or similar app on your phone.

Start the server and note the IP (e.g. http://10.14.25.228:8080/video).

Replace this line in the script with your phone’s IP:

URL = "http://<your_phone_ip>:8080/video"

2️⃣ Run the Script
python auto_face_recognition.py

🖼️ Workflow
Step 1: Face Capture

The script captures 30 images of your face.

Green box appears around detected faces.

Quit anytime with q.

Images are stored in auto_face_dataset/.

Step 2: Train Model

After capture, the script automatically trains an LBPH model.

Model is saved as auto_face_model.yml.

Step 3: Real-Time Recognition

Starts webcam recognition in real time.

Shows your face with:

Green box + label → Recognized face (with confidence score).

(Extendable) Red box → Unknown person (if you add multiple faces).
