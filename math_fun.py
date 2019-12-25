import numpy as np
import random


class Line():
    def __init__(self, k=0, b=0, p1=(0, 0), p2=(0, 0)):
        if p1==(0, 0) and p2==(0, 0):
            self.k = k
            self.b = b
        elif p2[0] - p1[0] != 0:
            self.k = (p2[1] - p1[1]) / (p2[0] - p1[0])
            self.b = p1[1] - self.k*p1[0]

    def y_value(self, x):
        return self.k*x + self.b

    def x_value(self, y):
        return (y - self.b)/self.k

    def cross_point(self, line):
        if self.k == line.k:
            print("Parallel and no cross point")
        else:
            p_x = (self.b - line.b) / (line.k - self.k)
            p_y = (self.k*line.b - line.k*self.b)/(self.k - line.k)
        return p_x, p_y

def OLS(point_list):
    lengh = point_list.shape[0]
    aver_x = np.mean(point_list[:,0])
    aver_y = np.mean(point_list[:,1])
    sum_Dx = 0
    sum_Dxy =0

    for i in range(lengh):
        sum_Dx += (point_list[i][0] - aver_x)**2
        sum_Dxy  += (point_list[i][0] - aver_x)*(point_list[i][1] - aver_y)

    a = sum_Dxy/sum_Dx
    b = aver_y - a*aver_x

    return a,b


def mid_point(p1, p2):
    p_x = (p1[0] + p2[0])/2
    p_y = (p1[1] + p2[1])/2
    return p_x, p_y

def line(x, a, b):
    y = a*x + b
    return y

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
