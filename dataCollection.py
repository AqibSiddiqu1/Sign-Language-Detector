import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math 
import time

cap= cv2.VideoCapture(0)#id number for our web camp , here we are capturing the image
detector = HandDetector(maxHands=1)#here we are using max hands 1 bcuz we want only 1 hand
offset=20# we are using offset bcuz when we are cropping the image its going out of the box
imgSize=300
folder= "Data/Aqib"
counter =0 

while True:
    success, img= cap.read()
    hands, img= detector.findHands(img)#here we are giving img to be detected  
    if hands:#here we are cropping the image
        hand=hands[0]#bcuz we have only have 1 hand thats we are intializing it with 0
        x,y,w,h=hand['bbox']  # xaxis yaxis width and height bbox:bounding box
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255#here we are creating a matrix using numpy (width,height)it will form a square .uint8 it means unsigned interger of 8 bit,we are multiplying it by 255 bcuz the image white is coming out to be black
        imgCrop=img[y-offset:y+h+offset,x-offset:x+w+offset]#its a matrix thats why we are defining y is the starting height starting width x and ending width x+w
        imgCropShape = imgCrop.shape
        
        aspectRatio=h/w  #if the value is 1 it means height is greater 
        if aspectRatio>1: 
            k=imgSize/h#here k is constant 
            wCal=math.ceil(k*w)#its the width calculated,w is previous width,ceil fuction allows to round off the values (higher side)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)#wgap is the width gap to push the image to center in imgwhite
            imgWhite[:,wGap:wCal+wGap] =imgResize #here we are putting the crop image on the white matrix 
        else:
            k=imgSize/w 
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:] =imgResize 
        
        cv2.imshow("ImageCrop", imgCrop)#it will form another window with cropped image
        cv2.imshow("ImageWhite", imgWhite)
        

    cv2.imshow("Image", img)# here wee are showing our image
    key = cv2.waitKey(1)#it will give a 1 mili second delay 
    if key == ord("s"):#s key for clicking
        counter +=1# here we are calculating how many image we have clicked
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)#time.time will give a unique value
        print(counter)