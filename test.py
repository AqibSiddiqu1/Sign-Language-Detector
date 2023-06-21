import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)  # Initialize video capture from default camera
detector = HandDetector(maxHands=1)  # Initialize hand detector
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")  # Initialize the classifier

offset = 20  # Offset for cropping the hand region
imgSize = 300  # Size of the output image

counter = 0  # Counter for the number of images captured

labels = ["A", "B", "C", "Hi", "Aqib", "victory"]  # List of labels for classification

while True:
    success, img = cap.read()  # Read a frame from the video capture
    if img is None:
        break
    imgOutput = img.copy()  # Create a copy of the frame for drawing output
    hands, img = detector.findHands(img)  # Detect hands in the frame
    
    if hands:
        hand = hands[0]  # Get the first detected hand
        x, y, w, h = hand['bbox']  # Get the bounding box coordinates of the hand

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create a white image
        imgCrop = img[y - offset : y + h + offset, x - offset : x + w + offset]  # Crop the hand region
        imgCropShape = imgCrop.shape  # Get the shape of the cropped image

        aspectRatio = h / w  # Calculate the aspect ratio of the hand

        if aspectRatio > 1:  # If the aspect ratio is greater than 1, adjust the width
            k = imgSize / h  # Calculate the scaling factor
            wCal = math.ceil(k * w)  # Calculate the new width
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))  # Resize the cropped image
            imgResizeShape = imgResize.shape  # Get the shape of the resized image
            wGap = math.ceil((imgSize - wCal) / 2)  # Calculate the gap to center the image horizontally
            imgWhite[:, wGap : wCal + wGap] = imgResize  # Place the resized image on the white background
            prediction, index = classifier.getPrediction(imgWhite, draw=False)  # Classify the image
            print(prediction, index)
        else:  # If the aspect ratio is less than or equal to 1, adjust the height
            k = imgSize / w  # Calculate the scaling factor
            hCal = math.ceil(k * h)  # Calculate the new height
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize the cropped image
            imgResizeShape = imgResize.shape  # Get the shape of the resized image
            hGap = math.ceil((imgSize - hCal) / 2)  # Calculate the gap to center the image vertically
            imgWhite[hGap : hCal + hGap, :] = imgResize  # Place the resized image on the white background
            prediction, index = classifier.getPrediction(imgWhite, draw=False)  # Classify the image

        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset - 50),
            (x - offset + 90, y - offset - 50 + 50),
            (255, 0, 255),
            cv2.FILLED
        )  # Draw a filled rectangle as background for the label text
        cv2.putText(
            imgOutput,
            labels[index],
            (x, y - 26),
            cv2.FONT_HERSHEY_COMPLEX,
            1.7,
            (255, 255, 255),
            2
        )  # Put the predicted label text on the output image
        cv2.rectangle(
            imgOutput,
            (x - offset, y - offset),
            (x + w + offset, y + h + offset),
            (255, 0, 255),
            4
        )  # Draw a rectangle around the hand region

        cv2.imshow("ImageCrop", imgCrop)  # Display the cropped image
        cv2.imshow("ImageWhite", imgWhite)  # Display the final image with white background

    cv2.imshow("Image", imgOutput)  # Display the original image
    cv2.waitKey(1)  # Wait for a key press
