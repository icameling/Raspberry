#-*- coding: utf-8 -*-
'''
@time: 2016/3/15/22:55
'''
#-*- coding: utf-8 -*-
'''
@time: 2016/2/21/15:18
'''

import numpy as np
import cv2
import os,glob
import operator

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 3600


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
        if (self.intRectHeight/self.intRectWidth < 1.0) or (self.intRectHeight/self.intRectWidth > 4.0): return False
        return True

class knn():

    file_list = []

    dir_path = ["E:\\opencv3\\step3_more_imgs\\0",
                 "E:\\opencv3\\step3_more_imgs\\1",
                 "E:\\opencv3\\step3_more_imgs\\2",
                 "E:\\opencv3\\step3_more_imgs\\3",
                 "E:\\opencv3\\step3_more_imgs\\4",
                 "E:\\opencv3\\step3_more_imgs\\5",
                 "E:\\opencv3\\step3_more_imgs\\6",
                 "E:\\opencv3\\step3_more_imgs\\7",
                 "E:\\opencv3\\step3_more_imgs\\8",
                 "E:\\opencv3\\step3_more_imgs\\9"]

    extension_list = ["jpg"]
    knnn = cv2.ml.KNearest_create()
    strFinalString = ""         # declare final string, this will have the final number sequence by the end of the program

    def get_file_list(self,pic_dir):
        os.chdir(pic_dir)
        self.file_list = []
        for extension in self.extension_list:
            extension = '*.' + extension
            self.file_list += [os.path.realpath(e) for e in glob.glob(extension)]

    def prepare_knn(self):
        nFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        nClassifications = []

        for pic_dir in self.dir_path:
            self.get_file_list(pic_dir)
            for dir in self.file_list:
                print dir,
                print ord(pic_dir[-1])
                img = cv2.imread(dir,0)
                if img is None:
                    continue
                imgROIResized = cv2.resize(img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                nClassifications.append(ord(pic_dir[-1]))
                #nClassifications.append(chr(dir[-5]))
                flattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                nFlattenedImages = np.append(nFlattenedImages, flattenedImage, 0)

        #nFlattenedImages = np.array(nFlattenedImages, np.float32)
        fltClassifications = np.array(nClassifications, np.float32)
        npClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

        np.savetxt("classifications.txt", npClassifications)           # write flattened images to file
        np.savetxt("flattened_images.txt", nFlattenedImages)          #
        print "save finished"

    def train_knn(self):
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

        self.knnn = cv2.ml.KNearest_create()                   # instantiate KNN object
        self.knnn.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

        print "knn train finished.."


    def show_pic_char(self,frame):

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
        #cv2.drawContours(imgGray,npaContours,5,(0, 255, 0),2)
        #cv2.imshow("con",imgGray)
        #print npaContours
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

        validContoursWithData.sort(key = operator.attrgetter("intRectX"))         # sort contours from left to right

        self.strFinalString = ""

        imgThreshshow = imgThresh.copy()
        for contourWithData in validContoursWithData:            # for each contour
                                                    # draw a green rect around the current char
            cv2.rectangle(imgThreshshow,                                        # draw rectangle on original testing image
                          (contourWithData.intRectX, contourWithData.intRectY),     # upper left corner
                          (contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),      # lower right corner
                          (125, 200, 110),              # green
                          2)                        # thickness

            imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,     # crop char out of threshold image
                               contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]

            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))             # resize image, this will be more consistent for recognition and storage

            npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))      # flatten image into 1d numpy array

            npaROIResized = np.float32(npaROIResized)       # convert from 1d numpy array of ints to 1d numpy array of floats

            retval, npaResults, neigh_resp, dists = self.knnn.findNearest(npaROIResized, k = 1)     # call KNN function find_nearest

            strCurrentChar = str(chr(int(npaResults[0][0])))                                             # get character from results

            self.strFinalString = self.strFinalString + strCurrentChar            # append current char to full string

        #writeVal(strFinalString)
        cv2.imshow("frame", imgThreshshow)

    def print_char(self):
        print "ans = ",
        print self.strFinalString


def test_in_win7():
    k = knn()
    k.prepare_knn()
    k.train_knn()
    img = cv2.imread("E:\\opencv3\\python char recg\\test1.png")
    cv2.imshow("tt",img)
    k.show_pic_char(img)
    k.print_char()


def test_knn2():

    k = knn()
    for pic_dir in k.dir_path:
        k.get_file_list(pic_dir)
        for dir in k.file_list:
            print dir,

            info = dir.split('\\')
            num = info[dir.count('\\')]
            num = num[:len(num) - 4]
            print pic_dir[-1]

if __name__ == '__main__':

    #test_knn2()
    test_in_win7()
    while cv2.waitKey(5)&0xff != 27:
        pass
    print "finished...."

