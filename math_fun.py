import numpy as np
import random
import math
import time
#数学几何运算相关类与函数，包含椭圆和直线两种类


class Line2():      #采用点法式重写直线类
    def __init__(self, r='#', angle='#', p1=None, p2=None, k='#', b='#'):
        if r != '#' and angle != '#':
            self.r = r
            self.angle = angle
        else:
            if p1 and p2 and p2[0] == p1[0]:
                if p1[0] >= 0:
                    self.angle = 0
                else:
                    self.angle = math.pi
                self.r = abs(p1[0])
            else:
                if p1 and p2:
                    self.k = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    self.b = p1[1] - self.k*p1[0]
                elif k!='#' and b!='#':
                    self.k = k
                    self.b = b  
               # print(str(self.k)+','+str(self.b))                        
                if self.k == 0:
                    if self.b > 0:
                        self.angle = math.pi/2
                    else:
                        self.angle = -math.pi/2
                elif self.k != 0:
                    if self.b > 0:
                        if self.k > 0:
                            self.angle =  math.pi + math.atan(-(1/self.k)) 
                        elif self.k < 0:
                            self.angle =  math.atan(-(1/self.k))
                    elif self.b < 0:
                        if self.k > 0:
                            self.angle =  math.atan(-(1/self.k))
                        elif self.k < 0:
                            self.angle =  math.atan(-(1/self.k)) - math.pi  
                    elif self.b == 0:
                        self.angle = math.atan(-(1/self.k))
                self.r = abs(self.b)/math.sqrt(self.k**2 + 1)
        self.evident = 1


    def cross_point(self, line):      #求与另外直线的交点
        if self.angle == line.angle:
            #print("Parallel and no cross point")
            return None
        else:
            p_x = (math.sin(line.angle)*self.r - math.sin(self.angle)*line.r)/math.sin(line.angle - self.angle)
            p_y = (math.cos(line.angle)*self.r - math.cos(self.angle)*line.r)/math.sin(self.angle - line.angle)
           # print("cross point: "+'('+str(p_x)+", "+str(p_y)+')')
        return [p_x, p_y]         


    def similar(self, line, simity):
        if abs(self.angle - line.angle) > (1-simity)*(math.pi/2):
            return False
        if abs(self.r - line.r) > (1-simity)*(0.9*self.r + 90):
            return False
        return True


    def fuse(self, line):
        self.r = (self.r*self.evident + line.r)/(self.evident + 1)
        self.angle = (self.angle*self.evident + line.angle)/(self.evident + 1)
        self.evident += 1
        

    def fun(self, x):
        if self.angle == 0 or self.angle == math.pi:
            return ['x', self.r/math.cos(self.angle)]
        else:
            return ['y', (self.r - x*math.cos(self.angle))/math.sin(self.angle)]


    def print_formula(self):
        if abs(self.angle - 0) < 0.0000001 or abs(self.angle - math.pi) < 0.0000001:
            str_y = " "
            str_x = str(math.cos(self.angle)) + "*x"
        elif abs(self.angle - math.pi/2) < 0.0000001 or abs(self.angle + math.pi/2) < 0.0000001:
            str_x = " "
            str_y = str(math.sin(self.angle)) + "*y"
        else:
            str_x = " + ("+str(math.cos(self.angle)) + ")*x"
            str_y = str(math.sin(self.angle)) + "*y"
        print(str(self.r)+" = "+str_y+str_x)
        print(str(self.r)+" = sin("+str(self.angle)+")*y + cos("+str(self.angle)+")*x")


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
            #self.print_fomula()
            # self.angle = 0.5*math.atan(2*self.b/(self.a-self.c))   #椭圆长轴倾角公式
            try:
                self.long_axis = 2*(self.a*self.cen[0]**2+self.c*self.cen[1]**2\
                    +2*self.b*self.cen[0]*self.cen[1]-1) / (self.a+self.c\
                        +((self.a-self.c)**2+(2*self.b)**2)**0.5)           #长轴公式
                self.short_axis = 2*(self.a*self.cen[0]**2+self.c*self.cen[1]**2\
                    +2*self.b*self.cen[0]*self.cen[1]-1) / (self.a+self.c\
                        -((self.a-self.c)**2+(2*self.b)**2)**0.5)           #短轴公式
            except BaseException:
                print("a: "+str(self.a)+"b: "+str(self.b)+"c: "+str(self.c))
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
        # if abs(self.angle - a_oval.angle) > (1-simity)*self.angle:
        #     return False
        if abs(self.long_axis - a_oval.long_axis) > (1-simity)**2*self.long_axis:
            return False
        if abs(self.short_axis - a_oval.short_axis) > (1-simity)**2*self.short_axis:
            return False
        d_cen = (self.cen[0] - a_oval.cen[0])**2 + (self.cen[1] - a_oval.cen[1])**2
        if d_cen > (1-simity)**2*(self.long_axis+self.short_axis):
            return False

        return True    
        

    def fuse(self, a_oval):              #与新的椭圆进行融合，并更新相应参数和属性
        self.a =(self.a *self.evident + a_oval.a)/(self.evident + 1)   #全局的加权平均值
        self.b =(self.b *self.evident + a_oval.b)/(self.evident + 1)
        self.c =(self.c *self.evident + a_oval.c)/(self.evident + 1)
        self.cen[0] = (self.cen[0] *self.evident + a_oval.cen[0])/(self.evident + 1)
        self.cen[1] = (self.cen[1] *self.evident + a_oval.cen[1])/(self.evident + 1)

        # self.angle = 0.5*math.atan(2*self.b/(self.a-self.c))                 #椭圆长轴倾角公式
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
            if i > 1000:
                break
        self.points = points
        return np.array(points)


    def print_fomula(self):                               #打印椭圆方程
        if self.enable:
            print(self.lone_axis)
            print(self.short_axis)

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
    # print(sum_Dx)
    if sum_Dx/lengh < 1 :        #样本的方差若很小，则视为垂直于x轴
        if aver_x >= 0:
            angle = 0
        elif aver_x < 0:
            angle = math.pi
        r = abs(aver_x)
        line1 = Line2(r, angle)
    else:
        a = sum_Dxy/sum_Dx        
        b = aver_y - a*aver_x
        # print(str(a)+','+str(b))
        line1 = Line2(k=a, b=b)

    return line1


def OLS2(point_list):                      #输入点集，用最小二乘法的出直线，注意点集格式为 (y, x)
    n = point_list.shape[0]
    aver_x = np.mean(point_list[:,0])
    aver_y = np.mean(point_list[:,1])
    xy = point_list[:,0]*point_list[:,1]
    xx = point_list[:,0]*point_list[:,0]
    

    # print(np.sum(xy))
    # print(n*aver_x*aver_y)
    up = np.sum(xy)-n*aver_x*aver_y
    down= np.sum(xx)-n*aver_x**2
    # print(up)
    # print(down)
    if down == 0 :
        if aver_x >= 0:
            angle = 0
        elif aver_x < 0:
            angle = math.pi
        r = abs(aver_x)
        line1 = Line2(r, angle)
    else:
        a = up/down        
        b = aver_y - a*aver_x
        # print(str(a)+','+str(b))
        line1 = Line2(k=a, b=b)

    return line1



def mid_point(p1, p2):             #求中点
    p_x = (p1[0] + p2[0])/2
    p_y = (p1[1] + p2[1])/2
    return [p_x, p_y]


def get_mid(array):          #求中位数
    size = array.size
    array = np.sort(array)
    midnum = int(size/2)
    if size % 2 == 0:
        mid = int((array[midnum-1] + array[midnum])/2)
    elif size % 2 ==1:
        mid = array[midnum]  
    return mid


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
    points = []
    for i in range(1000):
        y = line(i, a, b) + random.randint(-5,5)
        points.append([i,y])
    return np.array(points)


def points_for_test_2(r, sita):
    if sita!=0 and sita!=math.pi:
        return points_for_test(-1/math.tan(sita), r/math.sin(sita))
    else:
        print("x=?")
        points = []
        real_x = r/math.cos(sita)

        for i in range(2000):
            x = real_x +random.randint(-1,0)
            points.append([real_x, i])
        points.append([real_x+1,-2])
        points.append([real_x-1,-1])
        return np.array(points)


def solo_equal(a, b, c):      #解一元二次方程
    delta = b**2 - 4*a*c
    if delta < 0:
        #print("there is no anwser")
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

def check_rectangle(lines, error=0.1745):        #检测直线集的直线能否围成一个矩形，erro是所能允许的误差
    nl = len(lines)
    if nl<4:                              #三条线可构不成矩形
        return False  
    para_lines = []             
    rect_lines = []   
    for i in range(nl):                 #寻找平行线集，两条一对
        for j in range(i+1,nl):
            if abs(lines[i].angle-lines[j].angle)%math.pi < error:
                para_lines.append([lines[i],lines[j]])
    n_pl = len(para_lines)
    if n_pl<2:                          #如果只有一对平行线，则不构成矩形
        return False

    for i in range(n_pl):             #在平行线集寻找垂直的关系，四条一组
        for j in range(i+1,n_pl):     #这里没有用平均值做差，可能会使误差变大，不过我也懒得改了
            if abs(abs(para_lines[i][0].angle-para_lines[j][0].angle)%math.pi-math.pi/2) < error:    
                rect_lines.append([para_lines[i], para_lines[j]])

    if len(rect_lines):
        return rect_lines
    else:
        return False


