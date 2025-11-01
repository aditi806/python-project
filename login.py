import cv2
import face_recognition
import pickle
import numpy as np

# Load the saved face data
with open("face_data.pkl", "rb") as f:
    data = pickle.load(f)

known_name = list(data.keys())[0]
known_encoding = list(data.values())[0]

# Start camera
cam = cv2.VideoCapture(0)
print("Show your face to the camera...")
ret, frame = cam.read()
cam.release()
cv2.destroyAllWindows()

# Detect and encode face from webcam
faces = face_recognition.face_locations(frame)
if faces:
    face_encoding = face_recognition.face_encodings(frame, faces)[0]

    # Compare with saved data
    distance = np.linalg.norm(face_encoding - known_encoding)

    if distance < 0.6:
        print(f"✅ Login Successful! Welcome {known_name}!")
    else:
        print("❌ Face not recognized.")
else:
    print("No face detected.")
