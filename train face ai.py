import os
import face_recognition
import numpy as np

# Path to the folder containing your training images
training_images_folder = "FOLDER_WHERE_YOUR_TRAINING_IMGS_ARE"

# List all image files in the folder
image_files = [file for file in os.listdir(training_images_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Initialize an empty list to store face encodings
face_encodings = []

# Generate face encodings for each image
for image_file in image_files:
    image_path = os.path.join(training_images_folder, image_file)
    print("Processing image:", image_path)
    image = face_recognition.load_image_file(image_path)
    face_encodings_temp = face_recognition.face_encodings(image)
    
    if face_encodings_temp:
        face_encodings.append(face_encodings_temp[0])
    else:
        print("No face detected in image:", image_path)

# Specify the full path to save the list of encodings
encodings_file_path = os.path.join(training_images_folder, "face_encodings.npy")
with open(encodings_file_path, "wb") as f:
    np.save(f, face_encodings)

print("Face encodings saved to:", encodings_file_path)
