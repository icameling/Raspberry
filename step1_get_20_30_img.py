#-*- coding: utf-8 -*-
'''
@time: 2016/2/14/19:53
'''

import  cv2
import numpy as np
import sys

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

def get_imgs():
    img = cv2.imread("E:\\opencv3\\python char recg\\training_chars.png")
    dir_pic_save = "E:\\opencv3\\python char recg\\pic5\\"
    if img is None:
        print "read img error"
        return
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray,(5,5),0)
    img_thresh = cv2.adaptiveThreshold(img_blurred,255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,# invert so foreground will be white, background will be black
                                       11,2)
    img_thresh_copy = img_thresh.copy()
    imgContours, ncontours, npaHierarchy = cv2.findContours(img_thresh_copy,
                                                              cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contours in ncontours:
        if cv2.contourArea(contours) > MIN_CONTOUR_AREA:
            [intX, intY, intW, intH] = cv2.boundingRect(contours)
            cv2.rectangle(img,(intX, intY),
                          (intX+intW,intY+intH),
                          (0, 0, 255),2)

            imgROI = img_thresh[intY:intY+intH, intX:intX+intW]
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
            cv2.imshow("img",img)
            cv2.imshow("imgROIResized",imgROIResized)
            intChar = cv2.waitKey(0)&0xff
            pic = str(intChar-48)

            if intChar == 27:                   # if esc key was pressed
                sys.exit()
            else:
                print dir_pic_save + pic + ".jpg"
                cv2.imwrite(dir_pic_save + pic + ".jpg",imgROIResized)








if __name__ == '__main__':

    get_imgs()
    print "finished...."
