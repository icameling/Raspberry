#-*- coding: utf-8 -*-
'''
@time: 2016/3/16/15:35
'''

import numpy as np
import cv2


IMAGE_WIDTH = 20
IMAGE_HEIGHT = 20
intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'),
                 ord('4'), ord('5'), ord('6'), ord('7'),
                 ord('8'), ord('9')]

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

def get17_pic():
    nums = [0,0,0,0,0,0,0,0,0,0]

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print "open camera error..."
        return 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        bin = cv2.medianBlur(bin, 3)
        _, contours, heirs = cv2.findContours( bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        try:
            heirs = heirs[0]
        except:
            heirs = []

        for cnt, heir in zip(contours, heirs):
            _, _, _, outer_i = heir
            if outer_i >= 0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (50 <= h <= 128  and w <= 3.0*h):
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
            if v_out.std() > 10.0:
                continue

            cv2.imshow('bin_roi', bin_roi)

            key = cv2.waitKey(3)
            if key == ord('q'):
                break
            if key == ord(' '):
                print "input a num:",
                key = cv2.waitKey(0)
                print key - ord('0'),
                if key in intValidChars:
                    num = key - ord('0')
                    nums[num] += 1
                    filename = dir_path[num]+"\\"+ str(nums[num])+".jpg"
                    print filename,
                    bin_roi_resized = cv2.resize(bin_roi, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    cv2.imwrite(filename, bin_roi_resized)
                    print "save ok.."

        cv2.imshow('frame', frame)
        cv2.imshow('bin', bin)

def test():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print "open camera error..."
        return 0

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 10)
        bin = cv2.medianBlur(bin, 3)

        _, contours, heirs = cv2.findContours( bin.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        try:
            heirs = heirs[0]
        except:
            heirs = []

        for cnt, heir in zip(contours, heirs):
            _, _, _, outer_i = heir
            if outer_i >= 0:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if not (16 <= h <= 128  and w <= 1.2*h):
                continue

            #让字体在图片中间，不至于拉伸变形
            pad = max(h-w, 0)
            x, w = x-pad/2, w+pad
            cv2.rectangle(frame, (x, y), (x+w, y+h), (20, 35, 187))
            bin_roi = bin[y:,x:][:h,:w]
            gray_roi = gray[y:,x:][:h,:w]

            #有一定量的像素, m为true/false表
            m = (bin_roi != 0)
            if not 0.1 < m.mean() < 0.4:
                continue

            #计算数组的标准差
            v_in, v_out = gray_roi[m], gray_roi[~m]
            if v_out.std() > 10.0:
                continue

        cv2.imshow('frame', frame)
        cv2.imshow('bin', bin)
        ch = cv2.waitKey(1) & 0xFF
        if ch == 27:
            break



if __name__ == '__main__':
    #test()
    get17_pic()
    print "finished...."
