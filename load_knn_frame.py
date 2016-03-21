#-*- coding: utf-8 -*-
'''
@time: 2016/2/21/13:11
'''
import cv2
import numpy as np
import operator
import os,glob
import serial


global knn
global frame
global cap

MIN_CONTOUR_AREA = 600
MAX_CONTOUR_AREA = 3600

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

class ContourWithData():

    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        if self.fltArea > MAX_CONTOUR_AREA: return False
        if (self.intRectHeight/self.intRectWidth < 1.0) or (self.intRectHeight/self.intRectWidth > 6.0): return False
        return True

def get_file_list(dir_path, extension_list):
    os.chdir(dir_path)
    file_list = []
    for extension in extension_list:
        extension = '*.' + extension
        file_list += [os.path.realpath(e) for e in glob.glob(extension) ]
    return file_list

def train_knn():
    pic_dir = ["E:\\opencv3\\python char recg\\pic1",
               "E:\\opencv3\\python char recg\\pic2",
               "E:\\opencv3\\python char recg\\pic3",
               "E:\\opencv3\\python char recg\\pic4",
               "E:\\opencv3\\python char recg\\pic5"]
    pic_list = ["jpg"]

    nFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    nClassifications = []

    for dir in pic_dir:
        file = get_file_list(dir, pic_list)
        for dir in file:
            print dir,
            print ord(dir[-5])
            img = cv2.imread(dir,0)
            if img is None:
                continue
            imgROIResized = cv2.resize(img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            nClassifications.append(ord(dir[-5]))
            #nClassifications.append(chr(dir[-5]))
            flattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
            nFlattenedImages = np.append(nFlattenedImages, flattenedImage, 0)

    #nFlattenedImages = np.array(nFlattenedImages, np.float32)
    fltClassifications = np.array(nClassifications, np.float32)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

    np.savetxt("classifications.txt", npaClassifications)           # write flattened images to file
    np.savetxt("flattened_images.txt", nFlattenedImages)          #
    print "save finished"
    #knn = cv2.ml.KNearest_create()
    #knn.train(nFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)


def load_knn_train():
    global knn

    try:
        npaClassifications = np.loadtxt("classifications.txt", np.float32)                  # read in training classifications
    except:
        print "error, unable to open classifications.txt, exiting program\n"
        return

    try:
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)                 # read in training images
    except:
        print "error, unable to open flattened_images.txt, exiting program\n"
        return

    npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))       # reshape numpy array to 1d, necessary to pass to call to train

    knn = cv2.ml.KNearest_create()                   # instantiate KNN object
    knn.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    print "knn train finished.."


def show_pic_char():

    global frame
    #frame = cv2.imread("E:\\opencv3\\python char recg\\test1.png")
    if frame is None:
        print "error: image not read from file \n\n"
        return

    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)

    imgThresh = cv2.adaptiveThreshold(imgBlurred,                           # input image
                                      255,                                  # make pixels that pass the threshold full white
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,       # use gaussian rather than mean, seems to give better results
                                      cv2.THRESH_BINARY_INV,                # invert so foreground will be white, background will be black
                                      11,                                   # size of a pixel neighborhood used to calculate threshold value
                                      2)                                    # constant subtracted from the mean or weighted mean

    imgThreshCopy = imgThresh.copy()
    imgContours, npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,             # input image, make sure to use a copy since the function will modify this image in the course of finding contours
                                                 cv2.RETR_EXTERNAL,         # retrieve the outermost contours only
                                                 cv2.CHAIN_APPROX_SIMPLE)   # compress horizontal, vertical, and diagonal segments and leave only their end points

    allContoursWithData = []
    validContoursWithData = []
    for npaContour in npaContours:                             # for each contour
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)                                     # add contour with data object to list of all contours with data


    for contourWithData in allContoursWithData:                 # for all contours
        if contourWithData.checkIfContourIsValid():             # check if valid
            validContoursWithData.append(contourWithData)       # if so, append to valid contour list
        # end if
    # end for

    validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    for contourWithData in validContoursWithData:            # for each contour
                                                # draw a green rect around the current char
        cv2.rectangle(frame,                                        # draw rectangle on original testing image
                      (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                      (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                      (0, 255, 0),              # green
                      2)                        # thickness

        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                           contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = knn.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

        strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

        strFinalString = strFinalString + strCurrentChar            # append current char to full string
    # end for

    print strFinalString                  # show the full string
    #writeVal(strFinalString)

    cv2.imshow("frame", frame)

def open_camera(num):
    global cap
    cap = cv2.VideoCapture(num)
    if cap.isOpened():
        return cap
    else:
        print "open camera error..."
        return False


def get_frame():
    global frame
    global cap

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            return frame
        else:
            print "frame read error..."
            return False

def gogogo():
    global frame
    global cap
    global knn
    train_knn()
    cap = open_camera(0)
    while cv2.waitKey(5)&0xff != 27:
        frame = get_frame()
        show_pic_char()


if __name__ == '__main__':

    #train_knn()
    #load_knn_train()
    gogogo()
    print "finished...."
