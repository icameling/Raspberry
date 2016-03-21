#-*- coding: utf-8 -*-
'''
@time: 2016/3/15/23:46
'''

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

IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
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
    dir_path = ["E:\\opencv3\\python_char_recg\\0",
            "E:\\opencv3\\python_char_recg\\1",
            "E:\\opencv3\\python_char_recg\\2",
            "E:\\opencv3\\python_char_recg\\3",
            "E:\\opencv3\\python_char_recg\\4",
             "E:\\opencv3\\python_char_recg\\5",
             "E:\\opencv3\\python_char_recg\\6",
             "E:\\opencv3\\python_char_recg\\7",
             "E:\\opencv3\\python_char_recg\\8",
             "E:\\opencv3\\python_char_recg\\9"]

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
        nFlattenedImages =  np.empty((0, IMAGE_WIDTH * IMAGE_HEIGHT))
        nClassifications = []

        for pic_dir in self.dir_path:
            self.get_file_list(pic_dir)
            for dir in self.file_list:
                print dir,
                print ord(pic_dir[-1])
                img = cv2.imread(dir,0)
                if img is None:
                    continue
                imgROIResized = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                nClassifications.append(ord(pic_dir[-1]))
                #nClassifications.append(chr(dir[-5]))
                flattenedImage = imgROIResized.reshape((1, IMAGE_WIDTH * IMAGE_HEIGHT))
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

        if frame is None:
            print "error: image not read from file \n\n"
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        bin = cv2.medianBlur(bin, 3)
        _, contours, heirs = cv2.findContours( bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        try:
            heirs = heirs[0]
        except:
            heirs = []

        self.strFinalString = ""
        for cnt, heir in zip(contours, heirs):
            _, _, _, outer_i = heir
            if outer_i >= 0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (64 <= h <= 128  and w <= 3.0*h):
                continue

            #让字体在图片中间，不至于拉伸变形
            pad = max(h-w, 0)
            x, w = x-pad/2, w+pad
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
            bin_roi = bin[y:,x:][:h,:w]
            gray_roi = gray[y:,x:][:h,:w]

            #有一定量的像素
            m = (bin_roi != 0)
            if not 0.1 < m.mean() < 0.4:
                continue

            v_in, v_out = gray_roi[m], gray_roi[~m]
            #print "v_out.std() ",v_out.std(),
            if v_out.std() > 15.0:
                continue

            '''
            s = "%f, %f" % (abs(v_in.mean() - v_out.mean()), v_out.std())
            cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)
            '''

             # resize image, this will be more consistent for recognition and storage
            bin_roi_resized = cv2.resize(bin_roi, (IMAGE_WIDTH, IMAGE_HEIGHT))
            # flatten image into 1d float numpy array
            newcomer = np.float32(bin_roi_resized.reshape((1, IMAGE_WIDTH * IMAGE_HEIGHT)))

            ret, results, neighbours ,dists = self.knnn.findNearest(newcomer, k = 1)     # call KNN function find_nearest


            #if dists.mean() < 6500000:
            strCurrentChar = str(chr(int(results[0][0])))                                             # get character from results
            self.strFinalString = self.strFinalString + strCurrentChar            # append current char to full string

            cv2.putText(frame, strCurrentChar, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 0, 0), thickness = 1)
        #writeVal(strFinalString)
        cv2.imshow("frame", frame)
        cv2.imshow("bin", bin)

    def print_char(self):
        print "ans = ",
        print self.strFinalString




def test_in_win7():
    k = knn()
    k.prepare_knn()
    k.train_knn()
    img = cv2.imread("E:\\opencv3\\python_char_recg\\test1.png")
    cv2.imshow("tt",img)
    k.show_pic_char(img)
    k.print_char()


if __name__ == '__main__':

    #test_knn2()
    test_in_win7()
    while cv2.waitKey(5)&0xff != 27:
        pass
    print "finished...."


