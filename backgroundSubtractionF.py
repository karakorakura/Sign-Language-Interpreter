import numpy as np
import sys
import cv2
from Tkinter import *
from PIL import Image
from PIL import ImageTk
from scipy import stats
import copy
import algorithm.process_image as process_image

####################################################################################
#                               GLOBAL SETUP                                       #

frameNumber = 0
cameraNumber=1; #0 for others
saveFrame = False; # A boolean value tell if the frame is to be saved or not
winName1 = "Live feed"
winName2 = "Background subtraction"
cv2.startWindowThread()
# cv2.namedWindow(winName1)
videoReader = cv2.VideoCapture(cameraNumber);
debugMode = 1 # 0 disable  1 enable

######################################################################################

def backgroundDetect(buffer =None ):
    if buffer ==None:
        buffer=50
    buf = buffer;
    stride = 15;
    r = np.random.randint(0,stride,buf+1);
    print r
    t=0;

    start = 100;
    k = start+stride*buf;
    diff = np.zeros(1,np.float64);

    print "reading frames",
    ret,frame = videoReader.read();


    # cv2.imshow(winName1,frame)
    for i in range(k+1):
        ret,f = videoReader.read(); # read the next video frame
        sys.stdout.write("#");
        # cv2.imshow(winName1,f);
        if( i == 1):
            height,width = f.shape[:2];
            framesR = np.zeros(shape=(height,width,buf+1),dtype=np.float64);
            print framesR.shape
            framesG = np.zeros(shape=(height,width,buf+1),dtype=np.float64);
            framesB = np.zeros(shape=(height,width,buf+1),dtype=np.float64);
            f_1 = f;
            # diff(i) = 0;
        else:
            # diff(i) = sum(sum(abs(rgb2gray(f)-rgb2gray(f_1))>0.07))/(s(1)*s(2));
            f_1 = f;


        if (t<=buf and i>=start and r[t] == i%stride):
            # cv2.imshow('222',f);
            print t,
            framesR[:,:,t] = f[:,:,0];
            framesG[:,:,t] = f[:,:,1];
            framesB[:,:,t] = f[:,:,2];
            t = t + 1;


        # %foreground = step(foregroundDetector, frame);
    print " \n frame reading finished, Now you can come back here in front of camera\n"
    background = f;
    temp=stats.mode(framesR,2)[0];
    # print temp;
    print temp.shape;
    background[:,:,0] =temp[:,:,0]
    temp=stats.mode(framesG,2)[0];
    # print temp;
    print temp.shape;
    background[:,:,1] =temp[:,:,0]

    temp=stats.mode(framesB,2)[0];
    print temp.shape;
    background[:,:,2] =temp[:,:,0]
    # print temp;
    # background[:,:,0] =
    # background[:,:,1] = stats.mode(framesG)[0];
    # background[:,:,2] = stats.mode(framesB)[0];


    # cv2.imshow(winName2,background);
    return background

# ###########################

def removeFaces(frame,faceCascade=None,faceCascadePath=None):
    if faceCascadePath==None:
        faceCascadePath ='haar_face.xml'

    if faceCascade==None:
        face_cascade = cv2.CascadeClassifier(faceCascadePath)

    ####  detect faces
    faces = face_cascade.detectMultiScale(fgray, 1.3, 5)

    #unfinished

def getBinaryImage(frame,frameYCrCb,frameGRAY):
    frameRGB = frame

    subtractedFrameRGB = abs( frameRGB - backgroundFrameRGB )
    minBinaryRangeRGB = np.array([30,30,30], np.uint8)
    maxBinaryRangeRGB = np.array([245,245,245], np.uint8)
    binaryFrameRGB = cv2.inRange( subtractedFrameRGB, minBinaryRangeRGB,maxBinaryRangeRGB)
    cv2.imshow('test binaryFrameRGB', binaryFrameRGB)


    subtractedFrameYCrCb = abs( frameYCrCb - backgroundFrameYCrCb )
    minBinaryRangeY = np.array([7,2,2], np.uint8);maxBinaryRangeY = np.array([245,250,250], np.uint8)
    minBinaryRangeCrCb = np.array([3,3,3], np.uint8);maxBinaryRangeCrCb = np.array([245,245,245], np.uint8)
    binaryFrameY = cv2.inRange( subtractedFrameYCrCb, minBinaryRangeY,maxBinaryRangeY)
    binaryFrameCrCb = cv2.inRange( subtractedFrameYCrCb, minBinaryRangeCrCb,maxBinaryRangeCrCb)
    binaryFrameYCrCb = binaryFrameY + binaryFrameCrCb;
    cv2.imshow('test binaryFrameYCrCb', binaryFrameYCrCb)

    subtractedFrameGRAY = abs( frameGRAY - backgroundFrameGRAY )
    minBinaryRangeGRAY = np.array([40], np.uint8)
    maxBinaryRangeGRAY = np.array([245], np.uint8)
    binaryFrameGRAY = cv2.inRange( subtractedFrameGRAY, minBinaryRangeGRAY,maxBinaryRangeGRAY)
    cv2.imshow('test binaryFrameGRAY', binaryFrameGRAY)

    frameAdded = binaryFrameGRAY + binaryFrameYCrCb + binaryFrameRGB ;
    cv2.imshow('addition of all Binary Images',frameAdded)

    return frameAdded

def getDenoisedImage(frame):
    frameBlurred = cv2.medianBlur(frame, 15)
    # frameBlurred = cv2.medianBlur(frame, 11)
    cv2.imshow('after medianBlur',frameBlurred)
    return frameBlurred

def getSkinMaskedImage1(mask,frameYCrCb):
        # minRangeYCrCb = np.array([0,133,77], np.uint8)
        # minRangeYCrCb = np.array([255,173,127], np.uint8)
        # modified test skin colour
        minRangeYCrCb = np.array([16,133,77], np.uint8)
        maxRangeYCrCb = np.array([240,173,127], np.uint8)
        frameMasked   = cv2.bitwise_and(frameYCrCb,frameYCrCb,mask= mask)
        cv2.imshow('masked YcbCr',frameMasked)
        skinRegion = cv2.inRange(frameMasked,minRangeYCrCb,maxRangeYCrCb)
        cv2.imshow('skinRegion',skinRegion)





        return skinRegion

def thresholdOtsu(frame):
	ret,thresh1 = cv2.threshold(frame,75,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	return thresh1




def regionFilling(frame):
	#Have to Tune structuring element
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	dilation = cv2.dilate(frame,kernel,iterations = 1)
	img_bw = 255*(frame> 5).astype('uint8')
	#opencv.imshow("dilate",dilation)
	return dilation


def getSkinMaskedImage2(mask,frame):
    frameMasked   = cv2.bitwise_and(frame,frame,mask= mask)
    converted = cv2.cvtColor(frameMasked, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    # tuned settings
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    # apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    lowerBoundary = np.array([170,80,30],dtype="uint8")
    upperBoundary = np.array([180,255,250],dtype="uint8")
    skinMask2 = cv2.inRange(converted, lowerBoundary, upperBoundary)

    skinMask = cv2.addWeighted(skinMask,0.5,skinMask2,0.5,0.0)
    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.medianBlur(skinMask, 5)
    return skinMask



  ##############################################################################################
 #                                     MAIN                                                   #
##############################################################################################

print " prompt to start background subtraction Press any key \n Get out of the frame after pressing the key q"
keyStroke = 'n'
while keyStroke != 'y' and keyStroke != 'Y' and keyStroke != 'q' and keyStroke != 'Q':
    keyStroke = str(raw_input("press y/Y/q/Q to start background detection"));

backgroundFrame=backgroundDetect(buffer=20);
backgroundFrameRGB=cv2.flip(backgroundFrame,1);# flip like frames so subtraction can be even
backgroundFrameYCrCb = cv2.cvtColor(backgroundFrameRGB,cv2.COLOR_BGR2YCR_CB)
backgroundFrameGRAY = cv2.cvtColor(backgroundFrameRGB,cv2.COLOR_BGR2GRAY)


while True:

    ret,frame= videoReader.read()
    frame = cv2.flip(frame,1)
    frameYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    frameGRAY = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frameBinary = getBinaryImage(frame,frameYCrCb,frameGRAY)
    frameDenoised = getDenoisedImage(frameBinary)
    frameDenoised =  regionFilling(frameDenoised) ##### test optional stage
    frameSkinMasked1 = getSkinMaskedImage1(frameDenoised,frameYCrCb)
    frameSkinMasked2 = getSkinMaskedImage2(frameDenoised,frame)
    frameSkinMasked = frameSkinMasked1 + frameSkinMasked2
    cv2.imshow('skinmasked ++ ', frameSkinMasked)
    frameFilled = regionFilling(frameSkinMasked)
    frameSkin  =  cv2.bitwise_and(frame,frame,mask=frameFilled)
    cv2.imshow('skinmasked ++ RGB ', frameSkin)







    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

videoReader.release()
cv2.destroyWindow(winName1)
# cv2.destroyWindow(winName2)
