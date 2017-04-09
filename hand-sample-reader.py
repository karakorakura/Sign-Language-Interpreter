import numpy as np
import sys
import cv2
from Tkinter import *
from PIL import Image
from PIL import ImageTk
from scipy import stats
import copy
# import algorithm.process_image as process_image

####################################################################################
#                               GLOBAL SETUP                                       #

frameNumber = 0
cameraNumber=1; #0 for others

save = True; # A boolean value tell if the frame is to be saved or not
winName1 = "Live feed"
winName2 = "Background subtraction"
cv2.startWindowThread()
# cv2.namedWindow(winName1)
videoReader = cv2.VideoCapture(cameraNumber);
debugMode = 1 # 0 disable  1 enable

skip=0;
x1,x2,y1,y2=120,220,120,220
######################################################################################
def saveFrame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    global frameNumber;
    #cv2.imshow("hello",frame)
    if save:
            #sFrame = cv2.resize(frame, (100, 100)) # resize to 100 x 100 to save
        #cv2.imshow("hello",frame)
        sFrame=frame[y1:y2,x1:x2];
        gFrame=gray[y1:y2,x1:x2];
        cv2.imwrite("output/RGB/frame-"+str(frameNumber)+".png", sFrame)
        cv2.imwrite("output/GRAY/frame-"+str(frameNumber)+".png", gFrame)

        frameNumber = frameNumber + 1

def showFrame(frame):
    global skip
    frame = cv2.resize(frame, (432, 324))
    # crop the image to make it square
    frame = frame[:,54:378]
    frame1=deepcopy(frame)
    # filp the frame to create mirror image effect
    # frame = cv2.flip(frame, 1)
    cv2.rectangle(frame1, (x2, y2), (x1, y1), (255,0,0), 2)
    cv2.imshow('cropped', frame1)

    if skip<=25:
        skip+=1
    else:
        skip=0;
        saveFrame(frame);
    # cv2.window.after(10, showFrame)


  ##############################################################################################
 #                                     MAIN                                                   #
##############################################################################################
# print " prompt to start background subtraction Press any key \n Get out of the frame after pressing the key q"
keyStroke = 'n'
while keyStroke != 'y' and keyStroke != 'Y' and keyStroke != 'q' and keyStroke != 'Q':
    keyStroke = str(raw_input("press y/Y/q/Q to start hand sample detection"));

while True:

    ret,frame= videoReader.read()
    frame = cv2.flip(frame,1)


    # frame = cv2.GaussianBlur(frame,(5,5),2)
    cv2.imshow('frame,live',frame)
    showFrame(frame)






    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoReader.release()
cv2.destroyWindow(winName1)
# cv2.destroyWindow(winName2)
