import numpy as np 
import cv2
import math 
import os,sys
import time

#这个文件准备用来初始化餐盘的各项数据，以供主函数匹配
class Plate():                                   #这里使用对象来存储一般餐盘的信息
    def __init__(self, form_str, form, color):    #初始化参数有形状代号，形状参数，颜色
        self.form_str = form_str
        self.form = form
        self.color = color
        if form_str=='round':
            self.size = math.pi*form**2
        elif form_str=='oval':
            self.size = math.pi*form[0]*form[1]
        elif form_str=='rectangle':
            self.size = form[0]*form[1]
        else:
            self.size = 0

class St_Plate(Plate):                               #标准的餐盘继承一般餐盘的所有属性，并多了一个价格属性
    def __init__(self, form_str, form, color,prize):
        super().__init__(form_str, form, color)
        self.prize = prize

st_plate_1 = St_Plate('round',70,(248,248,248),1)        #以毫米为单位
st_plate_2 = St_Plate('round',85,(248,248,248),2)        #以毫米为单位
st_plate_3 = St_Plate('rectangle',[70,70],(248,248,248),3)        #以毫米为单位
st_plate_4 = St_Plate('rectangle',[85,85],(248,248,248),4)        #以毫米为单位
st_plate_5 = St_Plate('oval',(5,8),(248,248,248),5)        #以毫米为单位
st_plates = [st_plate_1,st_plate_2,st_plate_3,st_plate_4,st_plate_5]