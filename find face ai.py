import cv2
import face_recognition
import numpy as np

# Load the saved encodings
with open("face_encodings.npy", "rb") as f:
    saved_face_encodings = np.load(f)

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # 0 refers to the default camera

while True:
    # Capture a frame from the camera
    ret, frame = video_capture.read()

    # Convert the BGR frame to RGB (required by face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # No face found, continue to the next frame
    if not face_locations:
        continue

    # Extract face encodings from the current frame
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Check if any of the recognized faces match your saved encodings
    for face_encoding in face_encodings:
        results = face_recognition.compare_faces(saved_face_encodings, face_encoding)
        if any(results):
            print("Welcome, it's you!")
        else:
            print("Sorry, I don't recognize you.")

    # Display the frame with face rectangles
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()
