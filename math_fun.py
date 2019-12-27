import numpy as np
import random
import math

class Line():        #自写线类
    def __init__(self, k=0, b=0, p1=[0, 0], p2=[0, 0]):
        if p1==[0, 0] and p2==[0, 0]:
            self.k = k
            self.b = b
        elif p2[0] - p1[0] != 0:
            self.k = (p2[1] - p1[1]) / (p2[0] - p1[0])
            self.b = p1[1] - self.k*p1[0]
        elif p2[0] - p1[0] == 0:
            self.k = 65535
            self.b = p1[1] - self.k*p1[0]

    def y_value(self, x):             #给定x，求y值
        return self.k*x + self.b

    def x_value(self, y):             #给定y，求x值
        return (y - self.b)/self.k

    def cross_point(self, line):      #求与另外直线的交点
        if self.k == line.k:
            print("Parallel and no cross point")
            return None
        else:
            p_x = (self.b - line.b) / (line.k - self.k)
            p_y = (self.k*line.b - line.k*self.b)/(self.k - line.k)
            print("cross point: "+'('+str(p_x)+", "+str(p_y)+')')
        return [p_x, p_y]

class Oval():          #自写椭圆类
    def __init__(self, a=0, b=0, c=0,cen = None,
        p1 = None, p2 = None, p3 = None):    #参数与中心点初始化，或者中心点和三个椭圆上的点初始化
        if not(a==0 and b==0 and c==0):           
            self.a = a 
            self.b = b
            self.c = c
            self.cen = cen
            if a*c - b*b > 0:
                self.enable = True
            else:
                print("a, b, c worng")
                self.enable = False
        else:
            mux_abc = np.array([[(p1[0]-cen[0])**2, 2*(p1[0]-cen[0])*(p1[1]-cen[1]), (p1[1]-cen[1])**2],
                                [(p2[0]-cen[0])**2, 2*(p2[0]-cen[0])*(p2[1]-cen[1]), (p2[1]-cen[1])**2],
                                [(p3[0]-cen[0])**2, 2*(p3[0]-cen[0])*(p3[1]-cen[1]), (p3[1]-cen[1])**2]])
            solu = solu_equals(mux_abc, np.array([1,1,1]))
            if solu != None:
                self.a, self.b, self.c = solu
                self.enable = True
            else:
                self.enable = False
                print("Can't make the oval whese")
            self.cen = cen
        self.evident = 1
        
        if self.enable:  
            self.print_fomula()
            self.angle = 0.5*math.atan(2*self.b/(self.a-self.c))   #椭圆长轴倾角公式
            self.long_axis = 2*(self.a*self.cen[0]**2+self.c*self.cen[1]**2\
                +2*self.b*self.cen[0]*self.cen[1]-1) / (self.a+self.c\
                    +((self.a-self.c)**2+(2*self.b)**2)**0.5)           #长轴公式
            self.short_axis = 2*(self.a*self.cen[0]**2+self.c*self.cen[1]**2\
                +2*self.b*self.cen[0]*self.cen[1]-1) / (self.a+self.c\
                    -((self.a-self.c)**2+(2*self.b)**2)**0.5)           #短轴公式
            self.points = None           #椭圆的整数点集，等到要用再调用相应函数启用
        else:
            self.angle = 0
            self.lone_axis = 0
            self.short_axis = 0
    
    def check_point(self, p_test):        
        if self.enable == True:
            left = self.a*(p_test[0] - self.cen[0])**2 +\
                 2*self.b*(p_test[0] - self.cen[0])*(p_test[1] - self.cen[1]) +\
                   self.c*(p_test[1] - self.cen[1])**2
            return abs(left - 1)
        else:
            return None


    def similar(self, a_oval, simity):         #判断于另外一个椭圆是否相似，simity是给定的相似度      
        if abs(self.angle - a_oval.angle) > (1-simity)*self.angle:
            return False
        if abs(self.long_axis - a_oval.long_axis) > (1-simity)*self.long_axis:
            return False
        if abs(self.short_axis - a_oval.short_axis) > (1-simity)*self.short_axis:
            return False
        d_cen = (self.cen[0] - a_oval.cen[0])**2 + (self.cen[1] - a_oval.cen[1])**2
        if d_cen > (1-simity)**2*(self.long_axis+self.short_axis):
            return False

        return True    
        

    def fuse(self, a_oval):              #与新的椭圆进行融合，并更新相应参数和属性
        self.a =(self.a + a_oval.a)/2
        self.b =(self.b + a_oval.b)/2
        self.c =(self.c + a_oval.c)/2
        self.cen[0] = (self.cen[0] +a_oval.cen[0])/2
        self.cen[1] = (self.cen[1] +a_oval.cen[1])/2

        self.angle = 0.5*math.atan(2*self.b/(self.a-self.c))                 #椭圆长轴倾角公式
        self.lone_axis = 2*(self.a*self.cen[0]**2+self.c*self.cen[1]**2\
            +2*self.b*self.cen[0]*self.cen[1]-1) / (self.a+self.c\
                +((self.a-self.c)**2+(2*self.b)**2)**0.5)                    #长轴公式
        self.short_axis = 2*(self.a*self.cen[0]**2+self.c*self.cen[1]**2\
                    +2*self.b*self.cen[0]*self.cen[1]-1) / (self.a+self.c\
                    -((self.a-self.c)**2+(2*self.b)**2)**0.5)                #短轴公式
        self.evident += 1


    def points_on_oval(self, thick=1):      #获取椭圆上的所有的整数点
        if self.enable == False:
            return None
        points = []
        flag_l = True
        flag_r = True
        i = 0
        while flag_l or flag_r:
            if flag_r:
                solo_r = solo_equal(self.c, self.b*2*i, self.a*i**2-1)
                if solo_r:
                    y_r1 = int(solo_r[0] + self.cen[1])
                    y_r2 = int(solo_r[1] + self.cen[1])
                    points.append([int(self.cen[0]) + i, y_r1])
                    points.append([int(self.cen[0]) + i, y_r2])
                else:
                    flag_r = False
            if flag_l:
                solo_l = solo_equal(self.c, -self.b*2*i, self.a*i**2-1)
                if solo_l:
                    y_l1 = int(solo_l[0] + self.cen[1])
                    y_l2 = int(solo_l[1] + self.cen[1])
                    points.append([int(self.cen[0]) - i, y_l1])
                    points.append([int(self.cen[0]) - i, y_l2])
                else:
                    flag_l = False
            i += 1
        self.points = points
        return np.array(points)


    def print_fomula(self):                               #打印椭圆方程
        if self.enable:
            print("oval: "+str(self.a)+'(x - ('+str(self.cen[0])\
                +"))^2 + 2*("+str(self.b)+")(x - ("+str(self.cen[0])\
                +"))(y - ("+str(self.cen[1])+")) + ("+str(self.c)\
                +")(y - ("+str(self.cen[1])+"))^2 = 1")



def OLS(point_list):                      #输入点集，用最小二乘法的出直线，注意点集格式为 (y, x)
    lengh = point_list.shape[0]
    aver_x = np.mean(point_list[:,0])
    aver_y = np.mean(point_list[:,1])
    sum_Dx = 0
    sum_Dxy =0

    for i in range(lengh):
        sum_Dx += (point_list[i][0] - aver_x)**2
        sum_Dxy  += (point_list[i][0] - aver_x)*(point_list[i][1] - aver_y)
    if sum_Dx == 0 :
        a = 65535
    else:
        a = sum_Dxy/sum_Dx
    b = aver_y - a*aver_x
    print("line is: y="+ str(a)+"x+("+str(b)+')')

    return [a,b]


def mid_point(p1, p2):             #求中点
    p_x = (p1[0] + p2[0])/2
    p_y = (p1[1] + p2[1])/2
    return [p_x, p_y]


def line(x, a, b):         #直线函数
    y = a*x + b
    return y

 
def solu_equals(matrix,consent):       #克拉默法则解线性方程组
    H, W = matrix.shape
    H_c = consent.shape[0]
    solo= []
    if H != H_c  or H != W:
        print("error equation")
        print(H,W,H_c)
        return None
    
    det_m = np.linalg.det(matrix)
   # print("det_m " +str(det_m))
    if det_m == 0:
        print("No solotion") 
        return None

    det_x = [0]*H
    for i in range(H):
        mux_i = matrix.copy()
        mux_i[:,i] = consent
    #    print(mux_i)
        det_x[i] = np.linalg.det(mux_i)
        solo.append(det_x[i]/det_m)
    
    return solo
    


def points_for_test(a, b):            #生成某直线附近的随机点来测试最小二乘法函数
    x = random.randint(0, 100)
    y = line(x, a, b) + random.randint(0,30)
    points = np.array([[x, y]])
    for i in range(1000):
        x = random.randint(0, 100)
        y = line(x, a, b) + random.randint(-5,5)
        points = np.vstack((points, np.array([[x, y]])))
    return points

def solo_equal(a, b, c):      #解一元二次方程
    delta = b**2 - 4*a*c
    if delta < 0:
        print("there is no anwser")
        return None
    elif delta == 0:
        x1 = -b/(2*a)
        x2 = -b/(2*a)
        return x1, x2
       # print("x1=x2=" + str(x1))
    else:
        x1 = (-b + math.sqrt(delta))/(2*a)
        x2 = (-b - math.sqrt(delta))/(2*a)
       # print("x1 = " + str(x1) + ',' + 'x2 = ' + str(x2))
        return x1, x2



