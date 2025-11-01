import cv2
import face_recognition
import pickle

name = input("Enter your name: ")

# Open your webcam
cam = cv2.VideoCapture(0)
print("Look at the camera... capturing your face.")
ret, frame = cam.read()
cam.release()
cv2.destroyAllWindows()

# Find and encode face
faces = face_recognition.face_locations(frame)
if faces:
    face_encoding = face_recognition.face_encodings(frame, faces)[0]

    # Save encoding with name
    data = {name: face_encoding}
    with open("face_data.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f"Face registered successfully for {name}!")
else:
    print("No face detected. Try again.")
