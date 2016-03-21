#-*- coding: utf-8 -*-
'''
@time: 2016/2/14/21:07
'''

import cv2
import numpy as np
import os,glob

MIN_CONTOUR_AREA = 100

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

    nFlattenedImages = np.array(nFlattenedImages, np.float32)
    fltClassifications = np.array(nClassifications, np.float32)
    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later

    print nFlattenedImages.dtype
    print npaClassifications.dtype

    kNearest = cv2.ml.KNearest_create()
    kNearest.train(nFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

    print "train finished.."
    frame = cv2.imread("E:\\opencv3\\python char recg\\pic1\\8.jpg",0)
    cv2.imshow("2",frame)
    strFinalString = ""

    ROIResized = cv2.resize(frame, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
    ROIfinal = ROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    ROIfinal = np.float32(ROIfinal)
    retval, npaResults, neigh_resp, dists = kNearest.findNearest(ROIfinal, k = 1)
    print retval, npaResults, neigh_resp, dists
    strCurrentChar = str(chr(int(npaResults[0][0])))
    strFinalString = strFinalString + strCurrentChar
    print strFinalString


if __name__ == '__main__':
    train_knn()
    print "finished...."
