import numpy as np
import random


class Line():
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

    def y_value(self, x):
        return self.k*x + self.b

    def x_value(self, y):
        return (y - self.b)/self.k

    def cross_point(self, line):
        if self.k == line.k:
            print("Parallel and no cross point")
            return None
        else:
            p_x = (self.b - line.b) / (line.k - self.k)
            p_y = (self.k*line.b - line.k*self.b)/(self.k - line.k)
            print("cross point: "+'('+str(p_x)+", "+str(p_y)+')')
        return [p_x, p_y]

class Oval():
    def __init__(self, a=0, b=0, c=0,cen = None,
        p1 = None, p2 = None, p3 = None):
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
            solo = solo_equal(mux_abc, np.array([1,1,1]))
            if solo != None:
                self.a, self.b, self.c = solo
                self.enable = True
            else:
                self.enable = False
                print("Can't make the oval whese")
            self.cen = cen
        self.evident = 1
    
    def check_point(self, p_test):
        if self.enable == True:
            left = self.a*(p_test[0] - self.cen[0])**2 +\
                 2*self.b*(p_test[0] - self.cen[0])*(p_test[1] - self.cen[1]) +\
                   self.c*(p_test[1] - self.cen[1])**2
            return abs(left - 1)
        else:
            return None


    def similar(self, a_oval, threshold):  #判断于另外一个椭圆是否相似，threshold是阈值      
        d_a = (self.a - a_oval.a)**2
        d_b = (self.b - a_oval.b)**2
        d_c = (self.c - a_oval.c)**2 
        d_cen = (self.cen[0] - a_oval.cen[0])**2 + (self.cen[1] - a_oval.cen[1])**2
        if d_cen <= threshold**2:
            return True
        else:
            return False
        

    def fuse(self, a_oval):
        self.a =(self.a + a_oval.a)/2
        self.b =(self.b + a_oval.b)/2
        self.c =(self.c + a_oval.c)/2
        self.cen[0] = (self.cen[0] +a_oval.cen[0])/2
        self.cen[1] = (self.cen[1] +a_oval.cen[1])/2
        self.evident += 1


    def print_fomula(self):
        if self.enable:
            print("oval: "+str(self.a)+'(x - ('+str(self.cen[0])\
                +"))^2 + 2*("+str(self.b)+")(x - ("+str(self.cen[0])\
                +"))(y - ("+str(self.cen[1])+")) + ("+str(self.c)\
                +")(y - ("+str(self.cen[1])+"))^2 = 1")
    

def OLS(point_list):      #输入点集，用最小二乘法的出直线，注意点集格式为 (y, x)
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


def mid_point(p1, p2):
    p_x = (p1[0] + p2[0])/2
    p_y = (p1[1] + p2[1])/2
    return [p_x, p_y]

def line(x, a, b):
    y = a*x + b
    return y


def solo_equal(matrix,consent):     #克拉默法则解方程组
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
    


def points_for_test(a, b):
    x = random.randint(0, 100)
    y = line(x, a, b) + random.randint(0,30)
    points = np.array([[x, y]])
    for i in range(1000):
        x = random.randint(0, 100)
        y = line(x, a, b) + random.randint(-5,5)
        points = np.vstack((points, np.array([[x, y]])))
    return points




# test_points =points_for_test(test_a, test_b)
# print(test_points.shape)
# out_a, out_b = OLS(test_points)
# print("y="+str(out_a)+'x+'+str(out_b))

# test_mux = np.array([[1,1,1],[2,-1,3],[4,1,9]])
# con = np.array([1,4,16])

# jie = solo_equal(test_mux, con)

# test_oval = Oval(cen = (0,0), p1=(0.5, 1.87), p2=(0,2), p3=(0.4, 1.92))
# print(test_oval.a, test_oval.b, test_oval.c)
# print(jie)