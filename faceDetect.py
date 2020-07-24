import cv2
import os
import numpy as np
from typing import Tuple, List

Array = List[List[float]]

def faceDetection(testImg: Array) -> Tuple:
    #Converting imgae to gray image
    grayImg = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)
    #Haarcascade frontalface feature
    _face = cv2.CascadeClassifier('haarcascade\haarcascade_frontalface_default.xml')
    #Detect Multiscale
    faces = _face.detectMultiScale(grayImg,scaleFactor=1.32,minNeighbors=5)
    return faces, grayImg
