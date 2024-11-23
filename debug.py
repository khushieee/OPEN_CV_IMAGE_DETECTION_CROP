# Import packages
import cv2
import numpy as np

# Read the image
img = cv2.imread("opencv.jpeg")

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Detect faces
faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3)

for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
    face = img[y:y + h, x:x + w]  # Extract the face region
    cv2.imshow('opencv.jpeg', face)

cv2.imshow("Detected Faces", img)
cv2.waitKey(0)
