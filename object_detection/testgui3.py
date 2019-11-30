import sys
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog
from PyQt5.uic import loadUi
from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
import traceback
import cv2
import os
import glob
import traceback
from collections import namedtuple
import numpy as np
import imutils
import cv2
import numpy as np
import numpy
import os,operator
import six.moves.urllib as urllib
import sys
import tensorflow as tf
from collections import defaultdict
from io import StringIO
import cv2
import time
from send_check import transmit_no

#import mysql.connector
from copy import copy
###############################################
'''
try:
    #conn = mysql.connector.connect(user='ahmed',password='a7med',host='173.194.111.57',database='test')

    conn = mysql.connector.connect(user='root',password='a7med',host='127.0.0.1',database='db-lp-ckpnt2019')
    print ("Database successfully connected")
except:
    print ("Please Check Your DB Details")
'''

###############################################

t0=time.time()

sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

import  requests, re,urllib
#############################################
class IPCamera:
    def __init__(self, url, SizePerReading, MaxFrameSize):
        self.url = url
        self.SizePerReading = SizePerReading
        self.MaxFrameSize = MaxFrameSize
        self.bytes = ''
        self.connection = None
    def get_frame(self):
        response = urllib.request.urlopen(self.url)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame
VideoUrl = 'http://192.168.43.1:8080//shot.jpg'#photoaf.jpg'
camera = IPCamera(VideoUrl, 4000, 100000)

MODEL_NAME = '60000'
#MODEL_NAME = ''

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
###############################################

################################################
MIN_CONTOUR_AREA = 200
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
MIN_CONTOUR_WIDTH = 15
MIN_CONTOUR_HEIGHT = 20
MAX_CONTOUR_WIDTH = 80
MAX_CONTOUR_HEIGHT = 80

class ContourWithData():

    # member variables ############################################################################
    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):
        if self.fltArea < MIN_CONTOUR_AREA or self.intRectWidth>self.intRectHeight: return False
        #if self.intRectWidth < MIN_CONTOUR_WIDTH or self.intRectHeight < MIN_CONTOUR_HEIGHT or self.intRectWidth > MAX_CONTOUR_WIDTH or self.intRectHeight > MAX_CONTOUR_HEIGHT: return False

        return True
#############################################
NUM_CLASSES = 1

#print ("mode done in ",time.time()-t0)
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
#print ("labeling done in ",time.time()-t0)
def load_image_into_numpy_array(image):
   # The function supports only grayscale images
    last_axis = -1
    dim_to_repeat = 2
    repeats = 3
    grscale_img_3dims = np.expand_dims(image, last_axis)
    training_image = np.repeat(grscale_img_3dims, repeats, dim_to_repeat).astype('uint8')
    assert len(training_image.shape) == 3
    assert training_image.shape[-1] == 3
    return training_image

########################################33
def transform(pos):
# This function is used to find the corners of the LP and the dimensions of the LP
    pts=[]
    n=len(pos)
    for i in range(n):
        pts.append(list(pos[i][0]))

    rect=order_points(pts)
    h1=np.sqrt((rect[0][0]-rect[2][0])**2 + (rect[0][1]-rect[2][1])**2)     #height of left side
    h2=np.sqrt((rect[1][0]-rect[3][0])**2 + (rect[1][1]-rect[3][1])**2)     #height of right side
    h=max(h1,h2)

    w1=np.sqrt((rect[0][0]-rect[1][0])**2 + (rect[0][1]-rect[1][1])**2)     #width of upper side
    w2=np.sqrt((rect[2][0]-rect[3][0])**2 + (rect[2][1]-rect[3][1])**2)     #width of lower side
    w=max(w1,w2)

    return int(w),int(h),rect
#########################################

def correc(tx):
    slist = list(tx)
    for i, c in enumerate(slist):
        if slist[i] == '5' and 0 <= i <= 3:  # only replaces semicolons in the first part of the text
            slist[i] = 'S'
        if slist[i] == '3' and 0 <= i <= 3:  # only replaces semicolons in the first part of the text
            slist[i] = 'B'
        if slist[i] == '6' and 0 <= i <= 3:  # only replaces semicolons in the first part of the text
            slist[i] = 'B'
        if slist[i] == '0' and 0 <= i <= 3:  # only replaces semicolons in the first part of the text
            slist[i] = 'O'
        if slist[i] == '8' and 0 <= i <= 3:  # only replaces semicolons in the first part of the text
            slist[i] = 'B'
        if slist[i] == '1' and 0 <= i <= 3:  # only replaces semicolons in the first part of the text
            slist[i] = 'I'
        if slist[i] == 'S' and 3 < i <= len(slist):  # only replaces semicolons in the first part of the text
            slist[i] = '5'
        if slist[i] == 'O' and 3 < i <= len(slist):  # only replaces semicolons in the first part of the text
            slist[i] = '0'
        if slist[i] == 'I' and 3 < i <= len(slist):  # only replaces semicolons in the first part of the text
            slist[i] = '1'
        if slist[i] == 'B' and 3 < i <= len(slist):  # only replaces semicolons in the first part of the text
            slist[i] = '8'
        if slist[i] == 'Z' and 3 < i <= len(slist):  # only replaces semicolons in the first part of the text
            slist[i] = '7'
        if slist[i] == 'T' and 3 < i <= len(slist):  # only replaces semicolons in the first part of the text
            slist[i] = '7'
        if slist[i] == 'P' and 3 < i <= len(slist):  # only replaces semicolons in the first part of the text
            slist[i] = '9'
        if slist[i] == 'C' and 3 < i <= len(slist):  # only replaces semicolons in the first part of the text
            slist[i] = '0'
        if slist[i] == 'Q' and 3 < i <= len(slist):  # only replaces semicolons in the first part of the text
            slist[i] = '0'
    s = ''.join(slist)
    return (s)
def order_points(pts):
  #input format
  #pts = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]

  #sort points by x
  tmp = sorted(pts, key=lambda point: point[0])

  #tmp[0] and tmp[1] is left point
  #determine, which is top, and which is bottom by y coordinate
  if tmp[0][1] > tmp[1][1]:
    tl = tmp[1]
    bl = tmp[0]
  else:
    tl = tmp[0]
    bl = tmp[1]

  #do it with right tmp[2] and tmp[3]
  if tmp[2][1] > tmp[3][1]:
    tr = tmp[3]
    br = tmp[2]
  else:
    tr = tmp[2]
    br = tmp[3]

  return np.array([tl,tr,bl,br])
# # Detection
npaClassifications = np.loadtxt("classifications1.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images1.txt", np.float32)
# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)#(16, 12)
#img_dir = "F://New folder//NIGHT//orig" # Enter Directory of all images



tres=time.time()

class gui1(QDialog):
    def __init__(self):
        super(gui1, self).__init__()
        loadUi('gui23.ui', self)

        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)

        #self.Approxi.clicked.connect(self.ApproxiClicked)
        #self.CannyButton.clicked.connect(self.CannyClicked)
        #self.Precise.clicked.connect(self.PreciseClicked)

    @pyqtSlot()
    def ApproxiClicked(self):
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:


                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                img = self.image  # cv2.imread("testimagesfinal//2.jpg")
                img=cv2.resize(img,(1280,960))

                img2=img.copy()
                image_np = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                image_np_expanded = np.expand_dims(image_np, axis=0)
                #self.displayImage(1)
                #(boxesz, scoresz, classesz, numz)=sess.run([detection_boxes, detection_scores, detection_classes, num_detections],feed_dict={image_tensor: image_np_expanded})
                #vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxesz),np.squeeze(classesz).astype(np.int32),np.squeeze(scoresz),category_index,use_normalized_coordinates=True,line_thickness=3)
                #destRGB = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                #self.image = destRGB
                #self.displayImage(1)
                #self.image = img
                #print(boxesz)






                tcam = time.time()
                # img=camera.get_frame()
                # cv2.imwrite("orig90909090.png",img)

                #print("acquisition  ends in ", time.time() - tcam)

                    ####################################################################################################
                td = time.time()

                rows = img.shape[0]
                cols = img.shape[1]
                inp = cv2.resize(img, (300, 300))
                inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

                # Run the model
                out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),

                                sess.graph.get_tensor_by_name('detection_scores:0'),
                                sess.graph.get_tensor_by_name('detection_boxes:0'),
                                sess.graph.get_tensor_by_name('detection_classes:0')],
                                feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

                # Visualize detected bounding boxes.

                (numz, scoresz,  boxesz, classesz)=out
                vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxesz),np.squeeze(classesz).astype(np.int32),np.squeeze(scoresz),category_index,min_score_thresh=0.6,use_normalized_coordinates=True,max_boxes_to_draw=50,
line_thickness=3)
                destRGB = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                self.image = destRGB
                self.displayImage(1)


                num_detections = int(out[0][0])
                # for i in range(num_detections):
                classId = int(out[3][0][0])
                score = float(out[1][0][0])
                bbox = [float(v) for v in out[2][0][0]]
                print(score)



                #print(sess.graph.get_tensor_by_name('num_detections:0'))
                if score > 0.6:
                    x = bbox[1] * cols #xmin = left
                    y = bbox[0] * rows #top  ymin=top
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    # cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                    #print ('left top')
                    #print (x,y,)
                    #print ('right bottom')
                    #print (right,bottom)

                    (im_height, im_width, _) = img.shape
                        # (xminn, xmaxx, yminn, ymaxx) = (x * im_width, y * im_width, bottom * im_height, right * im_height)
                        # cropped_image = tf.image.crop_to_bounding_box(inp, int(yminn), int(xminn),int(ymaxx - yminn), int(xmaxx - xminn))
                    cropped_image = tf.image.crop_to_bounding_box(img, int(y), int(x), int(bottom - y),
                                                                      int(right - x))

                    sess = tf.Session()
                    img_data = sess.run(cropped_image)
                    print("True LP")
                else:
                     print("no LP")
                    # continue

                #print("detection  ends in ", time.time() - td)
                self.image = img_data
                self.displayImage(2)
                cv2.imwrite("detected23.png",img_data)
    @pyqtSlot()
    def ExactClicked(self):
        image1 = self.image  # cv2.imread('new folder\\t6.png')
        #ratio = image1.shape[0] / 360.0  # 500
        #ratio=480/360
        orig = image1.copy()
        #image = imutils.resize(image1, height=250)  # 500
        #gray = cv2.bilateralFilter(image,9,75,75)
        image=cv2.resize(orig,(360,270)) #480x360  #320x240-> errors
        img2=image.copy()
        img3=image.copy()
        #cv2.imwrite("detectedresized.png",image)
        #gray = cv2.bilateralFilter(image,9,75,75)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bl=(1/273)*np.array(([1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]),dtype="int")
        #bl=(1/256)*np.array(([1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]),dtype="int")

        gray=cv2.filter2D(gray,-1,bl)

        self.image = cv2.Canny(gray, 20, 90)
        edged=self.image

        #self.image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1)

        self.displayImage(3)

        _,contours,_=cv2.findContours(edged.copy(),1,1)
        #cv2.drawContours(image,contours,-1,[0,255,0],2)
        #cv2.imshow('Contours',image)
        n=len(contours)
        max_area=0
        pos=0
        for i in contours:
            area=cv2.contourArea(i)
            if area>max_area:
                max_area=area
                pos=i
            #cv2.drawContours(img2,i,-1,[0,255,0],2) #all contours

            #all cont
            cv2.drawContours(img3,i,-1,[0,255,0],2)
        self.image=img3
        self.displayImage(4)
        #print(pos)
        #cv2.imwrite('largest.png',img2)
        #Orig largest contour
        cv2.drawContours(img2,pos,-1,[0,255,0],2)
        self.image=img2
        self.displayImage(5)
        self.image=image

        peri=cv2.arcLength(pos,True)
        approx=cv2.approxPolyDP(pos,0.04*peri,True)
        #Approx largest contour
        #cv2.drawContours(img2, [approx], -1, (0, 0, 255), 2)
        #self.image=img2
        #self.displayImage(5)
        self.image=image
        plate1=image.copy()


        #self.image=image
        #self.displayImage(4)


        w,h,arr=transform(approx)
        cv2.circle(plate1, (arr[0][0],arr[0][1]), 5, (0, 0, 255), -1)
        text = "A"
        cv2.putText(plate1, text, (arr[0][0],arr[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), lineType=cv2.LINE_AA)
        cv2.circle(plate1, (arr[1][0],arr[1][1]), 5, (0, 0, 255), -1)
        text = "B"
        cv2.putText(plate1, text, (arr[1][0],arr[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), lineType=cv2.LINE_AA)
        cv2.circle(plate1, (arr[2][0],arr[2][1]), 5, (0, 0, 255), -1)
        text = "C"
        cv2.putText(plate1, text, (arr[2][0],arr[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), lineType=cv2.LINE_AA)
        cv2.circle(plate1, (arr[3][0],arr[3][1]), 5, (0, 0, 255), -1)
        text = "D"
        cv2.putText(plate1, text, (arr[3][0],arr[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), lineType=cv2.LINE_AA)
        self.image=plate1
        self.displayImage(6)
        self.image=image
        pts2=np.float32([[0,0],[w,0],[0,h],[w,h]])
        pts1=np.float32(arr)


        M=cv2.getPerspectiveTransform(pts1,pts2)
        #print(M)
        plate=cv2.warpPerspective(image,M,(w,h))
        #print ("detection  ends in ",time.time()-tdet)
        plate = cv2.resize(plate, (250, 150))

        self.image = plate
        self.displayImage(7)

    @pyqtSlot()
    def seg(self):
        global npaFlattenedImages
        global npaClassifications



        imgTestingNumbers=self.image
        #cv2.imshow("Perspective Transform",plate ))
        #cv2.waitKey(0)
        imgGray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(imgGray, (5,5), 15)
                           # blur

        thresh1 = cv2.adaptiveThreshold(imgGray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,13,1)
        self.image=thresh1
        self.displayImage(8)

        gray=cv2.bilateralFilter(imgGray,21,75,75) #21 75 75
        plate1=gray

        #ret,thresh = cv2.threshold(gray,200,220,cv2.THRESH_BINARY_INV)
        thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,13,1)
        self.image=thresh
        self.displayImage(9)

            # cv2.imwrite ("test results//adaptthresh.png",imgThresh)

        ##########################################3

        comp, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        #print (output)
        charCandidates = np.zeros(thresh.shape, dtype="uint8")

        # loop over the unique components\\
        labelMask1 = np.zeros(thresh.shape, dtype="uint8")
        for label in np.unique(labels):



                # if this is the background label, ignore it
                if label == 0:
                    continue

                # otherwise, construct the label mask to display only connected components for the
                # current label, then find contours in the label mask
                labelMask = np.zeros(thresh.shape, dtype="uint8")
                labelMask[labels == label] = 255
                (_,cnts, _) = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # ensure at least one contour was found in the mask
                for c in cnts:
                    # grab the largest contour which corresponds to the component in the mask, then
                    # grab the bounding box for the contour
                    #c = max(cnts, key=cv2.contourArea)
                    a=cv2.contourArea(c)
                    (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)

                            # compute the aspect ratio, solidity, and height ratio for the component
                    aspectRatio = boxW / float(boxH)
                    #solidity = cv2.contourArea(c) / float(boxW * boxH)
                    heightRatio = boxH / float(plate1.shape[0])
                    #print(aspectRatio)

                            # determine if the aspect ratio, solidity, and height of the contour pass
                            # the rules tests
                    keepAspectRatio = 0.1<aspectRatio < 1.0
                    #keepSolidity = solidity > 0.20
                    keepHeight = 0.2 < heightRatio < 0.95
                    keepArea=a >= 200

                    # check to see if the component passes all the tests
                    if keepAspectRatio and  keepHeight and keepArea :

                        labelMask1+=labelMask


        allContoursWithData = []                # declare empty lists,
        validContoursWithData = []              # we will fill these sho


        npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))


        kNearest = cv2.ml.KNearest_create()
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

        thresh1=labelMask1
        cv2.imwrite("thres\\"+ "filte"+"-"+".jpg",thresh1)


        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh1, connectivity=8)
        #connectedComponentswithStats yields every seperated component with information on each of them, such as size
        #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
        sizes = stats[1:, -1]; nb_components = nb_components - 1

        # minimum size of particles we want to keep (number of pixels)
        #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
        min_size = 100

        #your answer image
        img2 = np.zeros((output.shape))
        #for every component in the image, you keep it only if it's above min_size
        for i in range(0, nb_components):

            if sizes[i] >= min_size:
                img2[output == i + 1] = 255
                img2=img2.astype(numpy.uint8)
                ###########################################
                #imgThreshCopy = imgThresh.copy()
        self.image=img2
        self.displayImage(10)

        imgThreshCopy = img2.copy()
        ###cv2.imwrite("final1.png",imgThreshCopy)
        imgThreshCopy=cv2.convertScaleAbs(imgThreshCopy)

        imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for npaContour in npaContours:
            contourWithData = ContourWithData()                                             # instantiate a contour with data object
            contourWithData.npaContour = npaContour                                         # assign contour to contour with data
            contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
            contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
            contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)
            allContoursWithData.append(contourWithData)

        for contourWithData in allContoursWithData:
            if contourWithData.checkIfContourIsValid() :
                        # check if valid
                validContoursWithData.append(contourWithData)


        validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right
        strFinalString = ""
        charNo=0
        #print ("Seg ends in ",time.time()-tseg)
        tocr=time.time()
        for contourWithData in validContoursWithData:
            charNo += 1
                    # for each contour
                    # draw a green rect around the current char
            cv2.rectangle(imgTestingNumbers,(contourWithData.intRectX, contourWithData.intRectY),(contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),(0, 0, 255),2)

            imgROI = imgThreshCopy[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
            cv2.imwrite("thres\\" + str(charNo) + ".jpg",imgROI )


            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

            npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

            retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k = 3)     # call KNN function find_nearest

            strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

            strFinalString = strFinalString + strCurrentChar
        self.image = imgTestingNumbers
        self.displayImage(11)
            # cv2.imwrite("test results//out" + str(charNo) + ".png",imgROI )
        text = strFinalString
        # tco=time.time()
        txx = correc(text)
        # print ("\n correction done in ",time.time()-tco)
        print("\n" + txx + "\n")
        self.Textlabel.setText(txx)

    @pyqtSlot()
    def loadClicked(self):
       fname, filter = QFileDialog.getOpenFileName(self, 'Open file', 'E://object_detection//testimages', "Image Files (*)")
       if fname:
           self.loadImage(fname)
       else:
           print('Invalid Image')

    @pyqtSlot()
    #def Txt(self,text11):
     #   self.Textlabel.setText("Hello World")  # text value

    def loadImage(self, fname):
       self.image = cv2.imread(fname)
       #self.displayImage(1)
       self.ApproxiClicked()
       self.ExactClicked()
       #self.largeCont()
       #self.Warpingf()
       self.seg()
       #self.Txt()

    def displayImage(self, window=1):
       qformat = QImage.Format_Indexed8

       if len(self.image.shape) == 3:

           if (self.image.shape[2]) == 4:
               qformat = QImage.Format_RGBA8888
           else:
               qformat = QImage.Format_RGB888
       img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
       img = img.rgbSwapped()
       if window == 1:
           self.imgLabel.setScaledContents(True)
           self.imgLabel.setPixmap(QPixmap.fromImage(img))
           self.imgLabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
       if window == 2:

           self.Approxi.setScaledContents(True)
           self.Approxi.setPixmap(QPixmap.fromImage(img))
           self.Approxi.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
       if window == 3:

           self.Cannylabel.setScaledContents(True)
           self.Cannylabel.setPixmap(QPixmap.fromImage(img))
           self.Cannylabel.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
       if window == 4:
           self.LargeCont.setScaledContents(True)
           self.LargeCont.setPixmap(QPixmap.fromImage(img))
           self.LargeCont.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
       if window == 5:
           self.fourpoints.setScaledContents(True)
           self.fourpoints.setPixmap(QPixmap.fromImage(img))
           self.fourpoints.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
       if window == 6:
           self.Precise.setScaledContents(True)
           self.Precise.setPixmap(QPixmap.fromImage(img))
           self.Precise.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
       if window == 7:
           self.presicethresh.setScaledContents(True)
           self.presicethresh.setPixmap(QPixmap.fromImage(img))
           self.presicethresh.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
       if window == 8:
           self.Bilateralfilt.setScaledContents(True)
           self.Bilateralfilt.setPixmap(QPixmap.fromImage(img))
           self.Bilateralfilt.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
       if window == 9:
           self.BFThresh.setScaledContents(True)
           self.BFThresh.setPixmap(QPixmap.fromImage(img))
           self.BFThresh.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
       if window == 10:
           self.ThreshFilter.setScaledContents(True)
           self.ThreshFilter.setPixmap(QPixmap.fromImage(img))
           self.ThreshFilter.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
       if window == 11:
           self.Seg.setScaledContents(True)
           self.Seg.setPixmap(QPixmap.fromImage(img))
           self.Seg.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)





app = QApplication(sys.argv)
window=gui1()
window.setWindowTitle('test')
window.show()
sys.exit(app.exec_())
