from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
#from keras.layers import Activation
from keras.optimizers import SGD
#from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
import numpy as np
import argparse
import cv2
import os
from os import listdir
import sys
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
print = sys.stdout.write
flush = sys.stdout.flush

def speakF(text):
    global engine,voices
    engine.setProperty('voice',voices[1].id)
    engine.say(text)
    engine.runAndWait()

def speakM(text):
    global engine,voices
    engine.setProperty('voice',voices[0].id)
    engine.say(text)
    engine.runAndWait()


print('Hi User, Good to have you here. This is me!, Sign Language Interpreter. I will try to interpret your Gestures, with Some Advanced Algorithms, working at my back ')
speakM('Hi User, Good to have you here. This is me!, Sign Language Interpreter. I will try to interpret your Gestures, with Some Advanced Algorithms, working at my back ')
print('\nHi User, I am your assistant S L I. I will tell you what characters being interpreted in Real Time ')
speakF('Hi User, I am your assistant S L I. I will tell you what characters being interpreted in Real Time ')




# Import libraries
#import os,cv2
#import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
#from sklearn.cross_validation import train_test_split

from keras import backend as K
# K.set_image_dim_ordering('th')
K.set_image_dim_ordering('tf')

#from keras.utils import np_utils
#from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam

num_classes = 26
num_epoch=30
num_channel=3
imgSize = (128,128)
num_classes = 10

def image_to_feature_vector(image, size=(64,64),flatten=True):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
    if flatten == False : return cv2.resize(image,size);
    return cv2.resize(image, size).flatten()


#%%
# Saving and loading model and weights
from keras.models import model_from_json
from keras.models import load_model
import h5py as h5py

#%%
##########################################################################################
# textGenerater
import ahocorasick
# dictionary = ['ok','i','am','human','good', 'morning','evening','day','night','sleep','work','how', 'are' , 'you','hello', 'what', 'how' , 'who' ,'is' , 'are' , 'was', 'when', 'you', 'time','1','2','3','4','5','try','trying' ];
# file1 =  open("word3000.txt","r")

tolerance = 5

file1 =  open("wordlist.txt","r")
dictionary=[]
for word in file1 :
    dictionary.append(word.strip('\n\r'))
#print (dictionary)
# make trie
A = ahocorasick.Automaton()
for idx, key in enumerate(dictionary):
   A.add_word(key, (idx, key))
# Now convert the trie to an Aho-Corasick automaton to enable Aho-Corasick search:
A.make_automaton()


# print (dictionary)

charStream = " ";
textStream =" ";
letter = " "
sentenceStream =" "
printedSentences = []
tupleResults = []
wordcnt =0;
skipwords = 4
hashtable = {'a':0,'aboard':0,'allgone':0,'arrest':0, 'b':0
                ,'beside':0,'c':0,'d':0,'delete':0,'e':0,'f':0
                ,'g':0,'h':0,'house':0,'hungry':0,'hunt':0
                ,'i':0,'j':0,'k':0, 'l':0,'listen':0
                ,'m':0,'man':0,'me':0
                ,'n':0,'o':0,'oath':0,'p':0,'prisoner':0
                ,'q':0, 'r':0
                ,'s':0,'t':0,'u':0,'v':0,'w':0,'x':0
                ,'y':0,'z':0}
def clear_hash():
    for key , value in hashtable.items():
        hashtable[key]=0

def check_in_hash():
    global tolerance;
    for key, value in hashtable.items():
        if value > tolerance:
            clear_hash()
            return key
    return ""


clear_hash();


def Text(inputCharacter):
    global A,tolerance,charStream,skipwords, textStream,wordcnt,letter, sentenceStream, printedSentences,tupleResults, hashtable;
#while(True):
#inputStream = input()
#for inputCharacter in inputStream:
    flush()
    print("\r TEXT STREAM "+textStream+"|||\t\t\t SENTENCE :"+sentenceStream+"\r")
    flush()
    flush()
    letterAddedFlag=0

#    inputCharacter = input();

    hashtable[inputCharacter]+=1

    label = check_in_hash();
    print (label)
    if len(label) ==1:
        letter = label
        textStream+=letter
        print("\t\t< "+letter+" >\t\t\t\t\t\t\t\t\t\t\t\t\t\n")
        # print("\r"+textStream+"\r")
        speakF(letter)
        letterAddedFlag=1
        charStream = inputCharacter
    elif label =='delete':
        speakF(textStream[-1])
        textStream = textStream[:-1]
        speakM("deleted")
    elif len(label) >1:
        print("\t\t\t\t\t----> "+label+"\t\t\t\t\t\t\t\t\t\t\t\t\t\n"),
        sentenceStream+=" "+label
        wordcnt+=1
        if wordcnt % skipwords == 0 : sentenceStream+=","
        speakM(label)
        # engine.runAndWait()


    if letterAddedFlag==1:
        # tupleResults = aho_corasick(textStream,dictionary)
        haystack = textStream
        tupleResults=[]
        for end_index, (insert_order, original_value) in A.iter(haystack):
            start_index = end_index - len(original_value) + 1
            tupleResults.append((start_index, end_index, (insert_order,  haystack[start_index:end_index+1])))
            # print((start_index, end_index, (insert_order, original_value)))
#            assert haystack[start_index:start_index + len(original_value)] == original_value
        # print (textStream)
        prevIndex=-1
        if len(tupleResults)>=4:
            prevstart_index, prevend_index, (previnsert_order, prevword) = (0,0,(0," "))
            tupleResults.sort()
            # print (tupleResults)
            prevIndex = -1
            for start_index, end_index, (insert_order, word) in tupleResults:
                if start_index==prevstart_index:
                    prevstart_index, prevend_index, (previnsert_order, prevword) = start_index, end_index, (insert_order, word)
                    continue;#skip
                # sentenceStream+=prevword
                # sentenceStream+=" "
                print("\t\t\t\t\t----> "+prevword+"\t\t\t\t\t\t\t\t\t\t\t\t\t\n"),
                sentenceStream+=" "+prevword
                wordcnt+=1
                if wordcnt % skipwords == 0 : sentenceStream+=","
                speakM(prevword)
                engine.runAndWait()
                break;

                # prevstart_index, prevend_index, (previnsert_order, prevword) = start_index, end_index, (insert_order, word)
            # sentenceStream+=prevword
            # sentenceStream+=" "
            # print(prevword),
            # tupleResults=[]
            haystack=haystack[prevend_index+1:]
            textStream = textStream[prevend_index+1:]


## last iteration
#prevstart_index, prevend_index, (previnsert_order, prevword) = (0,0,(0," "))
#tupleResults.sort()
## print (tupleResults)
#prevIndex = -1
#for start_index, end_index, (insert_order, word) in tupleResults:
#    if start_index==prevstart_index:
#        prevstart_index, prevend_index, (previnsert_order, prevword) = start_index, end_index, (insert_order, word)
#        continue;#skip
#    elif start_index<=prevend_index:
#        continue;#skip
#    sentenceStream+=prevword
#    sentenceStream+=" "
#    print(prevword),
#    prevstart_index, prevend_index, (previnsert_order, prevword) = start_index, end_index, (insert_order, word)
#sentenceStream+=prevword
#sentenceStream+=" "
#print(prevword),
#
#printedSentences.append(sentenceStream)
#sentenceStream = " "
#print(' ');
#textStream = textStream[prevIndex+1:]
#print (printedSentences)
#
#
#

###########################################################################################
#%%
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
model = loaded_model
print("Loaded model from disk\n")

# classes = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
classes = ['a','aboard','allgone','arrest', 'b'
                ,'beside','c','d','delete','e','f'
                ,'g','h','house','hungry','hunt'
                ,'i','j','k', 'l','listen'
                ,'m','man','me'
                ,'n','o','oath','p','prisoner'
                ,'q', 'r'
                ,'s','t','u','v','w','x'
                ,'y','z'
                ]

#%%
# Testing a new image
def predictImage(test_image):
#    test_image = cv2.imread(r'E:\CVProject\Sign-Language-to-Speech-master\Sign-Language-to-Speech-master\10NovemberTraining\Sign-Language-Interpreter-latest work\3\output3Ds\3.2.3ds.png')
    #test_image=cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image=cv2.resize(test_image,imgSize)
    test_image = np.array(test_image)
    test_image = test_image.astype('float32')
    test_image /= 255
    #print (test_image.shape)

    if num_channel==1:
    	if K.image_dim_ordering()=='th':
    		test_image= np.expand_dims(test_image, axis=0)
    		test_image= np.expand_dims(test_image, axis=0)
    #		print (test_image.shape)
    	else:
    		test_image= np.expand_dims(test_image, axis=3)
    		test_image= np.expand_dims(test_image, axis=0)
    #		print (test_image.shape)

    else:
    	if K.image_dim_ordering()=='th':
    		test_image=np.rollaxis(test_image,2,0)
    		test_image= np.expand_dims(test_image, axis=0)
    #		print (test_image.shape)
    	else:
    		test_image= np.expand_dims(test_image, axis=0)
    #		print (test_image.shape)

    # Predicting the test image
    #print((model.predict(test_image)))
    character = classes[model.predict_classes(test_image,verbose=0)[0]]
    return character

#%%


from pykinect2 import PyKinectV2
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectRuntime

import ctypes
import _ctypes
import pygame
import sys

import numpy as np
import cv2
from PIL import Image
from PIL import ImageTk
#from scipy import stats
import copy

if sys.hexversion >= 0x03000000:
    import _thread as thread
else:
    import thread

frameNumber = 403
frameNumberRead=0
SignLabel = "5"
# colors for drawing different bodies
SKELETON_COLORS = [pygame.color.THECOLORS["red"],
                  pygame.color.THECOLORS["blue"],
                  pygame.color.THECOLORS["green"],
                  pygame.color.THECOLORS["orange"],
                  pygame.color.THECOLORS["purple"],
                  pygame.color.THECOLORS["yellow"],
                  pygame.color.THECOLORS["violet"]]

def getDenoisedImage(frame):
    frameBlurred = cv2.medianBlur(frame, 7)
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

def getColorDefinedFrame(frame):
    frameYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    minRangeYCrCb = np.array([16,133,77], np.uint8)
    maxRangeYCrCb = np.array([240,173,127], np.uint8)
    # frameMasked   = cv2.bitwise_and(frameYCrCb,frameYCrCb,mask= mask)
    # cv2.imshow('masked YcbCr',frameMasked)
    mask = cv2.inRange(frameYCrCb,minRangeYCrCb,maxRangeYCrCb)
    mask = getDenoisedImage(mask)
    skinRegion = cv2.bitwise_and(frame,frame,mask= mask)
    cv2.imshow('skinRegion',skinRegion)
    return skinRegion








class BodyGameRuntime(object):
    def __init__(self):
        pygame.init()

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Set the width and height of the screen [width, height]
        self._infoObject = pygame.display.Info()
        self._screen = pygame.display.set_mode((self._infoObject.current_w >> 1, self._infoObject.current_h >> 1),
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

        pygame.display.set_caption("Kinect for Windows v2 Body Game")

        # Loop until the user clicks the close button.
        self._done = False

        # Used to manage how fast the screen updates
        self._clock = pygame.time.Clock()

        # Kinect runtime object, we want only color and body frames
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body)

        # back buffer surface for getting Kinect color frames, 32bit color, width and height equal to the Kinect color frame size
        self._frame_surface = pygame.Surface((self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height), 0, 32)

        # here we will store skeleton data
        self._bodies = None


    def draw_body_bone(self, joints, jointPoints, color, joint0, joint1):
        joint0State = joints[joint0].TrackingState;
        joint1State = joints[joint1].TrackingState;

        # both joints are not tracked
        if (joint0State == PyKinectV2.TrackingState_NotTracked) or (joint1State == PyKinectV2.TrackingState_NotTracked):
            return

        # both joints are not *really* tracked
        if (joint0State == PyKinectV2.TrackingState_Inferred) and (joint1State == PyKinectV2.TrackingState_Inferred):
            return

        # ok, at least one is good
        start = (jointPoints[joint0].x, jointPoints[joint0].y)
        end = (jointPoints[joint1].x, jointPoints[joint1].y)

        try:
            pygame.draw.line(self._frame_surface, color, start, end, 8)
        except: # need to catch it due to possible invalid positions (with inf)
            pass

    def draw_body(self, joints, jointPoints, color):
        # Torso
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Head, PyKinectV2.JointType_Neck);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_Neck, PyKinectV2.JointType_SpineShoulder);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_SpineMid);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineMid, PyKinectV2.JointType_SpineBase);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineShoulder, PyKinectV2.JointType_ShoulderLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_SpineBase, PyKinectV2.JointType_HipLeft);

        # Right Arm
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderRight, PyKinectV2.JointType_ElbowRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowRight, PyKinectV2.JointType_WristRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_HandRight);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandRight, PyKinectV2.JointType_HandTipRight);
       # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristRight, PyKinectV2.JointType_ThumbRight);

        # Left Arm
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ShoulderLeft, PyKinectV2.JointType_ElbowLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_ElbowLeft, PyKinectV2.JointType_WristLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_HandLeft);
        self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HandLeft, PyKinectV2.JointType_HandTipLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_WristLeft, PyKinectV2.JointType_ThumbLeft);

        # Right Leg
       # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipRight, PyKinectV2.JointType_KneeRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeRight, PyKinectV2.JointType_AnkleRight);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleRight, PyKinectV2.JointType_FootRight);

        # Left Leg
       # self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_HipLeft, PyKinectV2.JointType_KneeLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_KneeLeft, PyKinectV2.JointType_AnkleLeft);
        #self.draw_body_bone(joints, jointPoints, color, PyKinectV2.JointType_AnkleLeft, PyKinectV2.JointType_FootLeft);


    def draw_color_frame(self, frame, target_surface):
        target_surface.lock()
        address = self._kinect.surface_as_array(target_surface.get_buffer())
        ctypes.memmove(address, frame.ctypes.data, frame.size)
        del address
        target_surface.unlock()

    def draw_rectangle(self,frame):
        #rectangle
        winName  = "Rectangle"
        cv2.startWindowThread()
        cv2.namedWindow(winName)

        while True:
            cv2.imshow('live',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow(winName)

    def run(self):
        global frameNumber
        global frameNumberRead
        global SignLabel
        # -------- Main Program Loop -----------
        while not self._done:
            # --- Main event loop
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    self._done = True # Flag that we are done so we exit this loop

                elif event.type == pygame.VIDEORESIZE: # window resized
                    self._screen = pygame.display.set_mode(event.dict['size'],
                                               pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE, 32)

            # --- Game logic should go here

            # --- Getting frames and drawing
            # --- Woohoo! We've got a color frame! Let's fill out back buffer surface with frame's data
            if self._kinect.has_new_color_frame():
                frame = self._kinect.get_last_color_frame()
                frame1 = copy.deepcopy(frame)




                # frameNumber+=1
#                print(frameNumber)
#                print (frame.shape)
#                print(self._kinect.color_frame_desc.Width, self._kinect.color_frame_desc.Height)
                frame1=np.reshape(frame1,(1080,1920,4))
#                print (frame1.shape)
                winName  = "Rectangle"
                cv2.startWindowThread()
                cv2.namedWindow(winName)
                frame2 = frame1[:,:,0:4]
                frame3 = copy.deepcopy(frame2)


#                print(frame2.shape)
                # frameNumber
                # cv2.imwrite("/output1/frame-"+str(frameNumber)+".png", sFrame1)




                self.draw_color_frame(frame, self._frame_surface)


                frame = None

            # --- Cool! We have a body frame, so can get skeletons
            if self._kinect.has_new_body_frame():
                self._bodies = self._kinect.get_last_body_frame()

            # --- draw skeletons to _frame_surface
            if self._bodies is not None:
                for i in range(0, self._kinect.max_body_count):
                    body = self._bodies.bodies[i]
                    if not body.is_tracked:
                        continue

                    joints = body.joints
                    # convert joint coordinates to color space
                    joint_points = self._kinect.body_joints_to_color_space(joints)
                    self.draw_body(joints, joint_points, SKELETON_COLORS[i])

                    tiplx = int(joint_points[PyKinectV2.JointType_HandTipLeft].x)
                    tiply = int(joint_points[PyKinectV2.JointType_HandTipLeft].y)

                    tiprx = int(joint_points[PyKinectV2.JointType_HandTipRight].x)
                    tipry = int(joint_points[PyKinectV2.JointType_HandTipRight].y)

                    wristlx = int(joint_points[PyKinectV2.JointType_WristLeft].x)
                    wristly = int(joint_points[PyKinectV2.JointType_WristLeft].y)

                    wristrx = int(joint_points[PyKinectV2.JointType_WristRight].x)
                    wristry = int(joint_points[PyKinectV2.JointType_WristRight].y)

                    cv2.rectangle(frame2,(tiplx-100,tiply-100),(wristlx+100,wristly+100),(0,255,0),1)
                    cv2.rectangle(frame2,(tiprx-100,tipry-100),(wristrx+100,wristry+100),(0,0,255),1)



                    saveFrame = True
                    min_xr = min(tiprx,wristrx)
                    max_xr = max(tiprx,wristrx)
                    min_yr = min(tipry,wristry)
                    max_yr = max(tipry,wristry)

                    min_xl = min(tiplx,wristlx)
                    max_xl = max(tiplx,wristlx)
                    min_yl = min(tiply,wristly)
                    max_yl = max(tiply,wristly)

                    if max_xl + 100 > min_xr - 100 and ( (max_yl+100 > min_yr-100 and max_yl +100< max_yr+100) or (min_yl-100 < max_yr+100 and min_yl > min_yr) ):
                        minx_final = min(min_xl,min_xr) +10;
                        maxx_final = max(max_xl,max_xr) -10;
                        miny_final = min(min_yl,min_yr) +10;
                        maxy_final = max(max_yl,max_yr) -10;

                    else :
                        minx_final = min_xr
                        maxx_final = max_xr
                        miny_final = min_yr
                        maxy_final = max_yr

                    cv2.rectangle(frame2,(minx_final-100,miny_final-100),(maxx_final+100,maxy_final+100),(255,0,0),3)
                    frameToShow = cv2.resize(frame2,(960,540));
                    ### add text to image
                    font                   = cv2.FONT_HERSHEY_PLAIN
                    bottomLeftCornerOfText = (10,25)
                    fontScale              = 2
                    fontColor              = (25,21,85)
                    lineType               = 1
                    thickness              = 1


                    textFrame = np.zeros(shape=(840,640))


                    cv2.putText(textFrame,"TEXT STREAM || ",
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
                    cv2.putText(textFrame,""+textStream,
                        (10,55),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

                    cv2.putText(textFrame,"WORDS || "+"",
                        (10,100),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

                    y0, dy = 150, 18
                    for i, line in enumerate(sentenceStream.split(',')):
                        y = y0 + i*dy
                        cv2.putText(textFrame, line,
                            (10, y ),
                            font,
                            fontScale ,
                            fontColor,
                            lineType)


                    cv2.imshow("text",textFrame)
                    cv2.imshow(winName,frameToShow)


                    #print(min_x,max_x,min_y,max_y)
                    frameNumberRead = frameNumberRead + 1
                    if saveFrame and frameNumberRead%15==0:

                        if minx_final-100>0 and miny_final-100>0:

                            framesave = frame3[miny_final-100:maxy_final+100,minx_final-100:maxx_final+100,:]
#                            print(tiprx-100,wristrx+100,tipry-100,wristry+100)
                            sFrame = framesave
                            #sFrame = cv2.resize(framesave, (200, 200))
#                            print (sFrame.shape)
                            sFrame1 = framesave[:,:,0:3]
                            sFrame1s = getColorDefinedFrame(sFrame1)
#                            print (sFrame1.shape)

                            # name1 = r"F:\sem 6\capstone\Divanshu\Sign-Language-to-Speech-master\Live-feed-analyzer\output\frame-"+str(frameNumber)+".tiff"
                            # name2= r"F:\sem 6\capstone\Divanshu\Sign-Language-to-Speech-master\Live-feed-analyzer\output1\frame-"+str(frameNumber)+".png"
                            # SignLabel = "b"

#                            name4D = r".\output4D\%s.%s.%s"%( SignLabel ,  str(frameNumber) ,"4d.tiff")
#                            name3D = r".\output3D\%s.%s.%s"%( SignLabel ,  str(frameNumber) ,"3d.png")
#                            name3Ds = r".\output3Ds\%s.%s.%s"%( SignLabel ,  str(frameNumber) ,"3ds.png")
                            #
                            # name4D = r".\output4D\"   + SignLabel + r"." +  str(frameNumber) + "."+"4d.tiff"
                            # name3D = r".\output3D\"   + SignLabel + r"." + str(frameNumber) + "."+"3d.png"
                            # name3Ds = r".\output3Ds\" + SignLabel + r"." + str(frameNumber) + "."+"3ds.png"

#                            print(name4D,name3D,name3Ds)
                         # resize to 100 x 100 to save
                        # sFrame = cv2.resize(final, (100, 100)) # resize to 100 x 100 to save
                            #cv2.imshow('hello',framesave)
                            #cv2.imshow('hello',sFrame1)
#                            cv2.imwrite(name4D, sFrame)
#                            cv2.imwrite(name3D, sFrame1)
#                            cv2.imwrite(name3Ds, sFrame1s)


                            character = predictImage(sFrame1s)
                            print (". "+character+"\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n\n")
                            flush()
                            Text(character)
                            frameNumber = frameNumber + 1



            # --- copy back buffer surface pixels to the screen, resize it if needed and keep aspect ratio
            # --- (screen size may be different from Kinect's color frame size)
            h_to_w = float(self._frame_surface.get_height()) / self._frame_surface.get_width()
            target_height = int(h_to_w * self._screen.get_width())
            surface_to_draw = pygame.transform.scale(self._frame_surface, (self._screen.get_width(), target_height));
            self._screen.blit(surface_to_draw, (0,0))
            surface_to_draw = None
            pygame.display.update()

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self._clock.tick(60)

        # Close our Kinect sensor, close the window and quit.
        self._kinect.close()
        pygame.quit()


__main__ = "Kinect v2 Body Game"
game = BodyGameRuntime();
game.run();
