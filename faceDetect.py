import cv2
import os
import numpy as np
from typing import Tuple, List

Array = List[List[float]]

def faceDetection(testImg: Array) -> Tuple:
    #Converting imgae to gray image
    grayImg = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)
    #Haarcascade frontalface feature
    _face = cv2.CascadeClassifier('\\Users\\HP\\Desktop\\FaceDetect\\haarcascade\\haarcascade_frontalface_default.xml')
    #Detect Multiscale
    faces = _face.detectMultiScale(grayImg,scaleFactor=1.3,minNeighbors=5)
    return faces, grayImg

def draw_rectangle(faces: Array, testImg: str) -> None:

    for (x,y,w,h) in faces:
        cv2.rectangle(testImg,(x,y),(x+w,y+h),(0,0,0),thickness=1)

if __name__ == '__main__':
    im = cv2.imread("hari.JPEG")
    f, g = faceDetection(im)
    draw_rectangle(faces=f, testImg=im)
    cv2.imshow('face detecting', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()