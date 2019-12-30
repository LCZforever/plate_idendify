import numpy as np 
import cv2
import math 
import os,sys
import time


 
class Plate():
    def __init__(self, form, size, color, other_sign=None):
        pass
def show(img, strs,live_time=1):       #显示图片，live_time为0为按任意键继续
    cv2.imshow(strs, img)
    cv2.waitKey(live_time)




