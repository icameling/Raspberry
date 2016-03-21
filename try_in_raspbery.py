#-*- coding: utf-8 -*-
'''
@time: 2016/2/21/13:14
'''

import cv2
import numpy as np
import os,glob

global frame
global cap

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


def test_knn():
    import knn_package2 as ka

    k = ka.knn()
    k.train_knn()
    img = cv2.imread("E:\\opencv3\\python char recg\\test1.png")
    cv2.imshow("tt",img)
    k.show_pic_char(img)
    k.print_char()

def test_in_raspberry():
    import knn_package3 as ka
    global frame
    global cap

    k = ka.knn()
    k.train_knn()

    cap = open_camera(0)
    while cv2.waitKey(5)&0xff != 27:
        frame = get_frame()
        k.show_pic_char(frame)
        k.print_char()



if __name__ == '__main__':
    #test1()
    #test_knn()

    test_in_raspberry()
    while cv2.waitKey(5)&0xff != 27:
        pass
    print "finished...."
