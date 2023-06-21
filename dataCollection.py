import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math 
import time

cap = cv2.VideoCapture(0)  # Initialize video capture from default camera
detector = HandDetector(maxHands=1)  # Initialize hand detector
offset = 20  # Offset for cropping the hand region
imgSize = 300  # Size of the output image
folder = "Data/Aqib"  # Folder to store captured images
counter = 0  # Counter for the number of images captured

while True:
    success, img = cap.read()  # Read a frame from the video capture
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
        else:  # If the aspect ratio is less than or equal to 1, adjust the height
            k = imgSize / w  # Calculate the scaling factor
            hCal = math.ceil(k * h)  # Calculate the new height
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))  # Resize the cropped image
            imgResizeShape = imgResize.shape  # Get the shape of the resized image
            hGap = math.ceil((imgSize - hCal) / 2)  # Calculate the gap to center the image vertically
            imgWhite[hGap : hCal + hGap, :] = imgResize  # Place the resized image on the white background
        
        cv2.imshow("ImageCrop", imgCrop)  # Display the cropped image
        cv2.imshow("ImageWhite", imgWhite)  # Display the final image with white background
        
    cv2.imshow("Image", img)  # Display the original image
    key = cv2.waitKey(1)  # Wait for a key press
    
    if key == ord("s"):  # If 's' is pressed, capture the image
        counter += 1  # Increment the counter
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)  # Save the image with a unique filename
        print(counter)  # Print the number of captured images
