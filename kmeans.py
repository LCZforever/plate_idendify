import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as nr
import matplotlib.path as mpath
from itertools import zip_longest
# 思考; 如何定初始的中心点，使迭代次数降低
# 这个文件是为了把k_means算法封装起来，形成一个函数
# 函数设计
# 输入：n乘以2的numpy矩阵，分成的类数m
# 返回：m个xxx乘以2的矩阵

# 计算每个两点之间的距离平方
def distance_sq(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    # print(str(x1)+','+str(y1))
    return abs(x1-x2)**2+abs(y1-y2)**2


def apart(point, cen_dict):    # 分类
    cens = []
    for key in cen_dict.keys():
        cens.append(cen_dict[key][-1])
    point_dict = {}
    num = 0
    for i in cens:
        num += 1
        name = 'center_' + str(num)
        point_dict[name] = np.array([False])
    for p in point:
        dt = []
        for cen in cens:
            dt.append(distance_sq(p, cen))
        min_no = min_dt(dt)[0]
        name = 'center_' + str(min_no)
        if point_dict[name].all():
            point_dict[name] = np.concatenate([point_dict[name], p])
        else:
            point_dict[name] = p
    for key in point_dict.keys():
        if  point_dict[key].all():
            point_dict[key] = point_dict[key].reshape(-1, 2)
    return point_dict


def min_dt(dt_list):
    num = 0
    min_num = dt_list[0]
    n = 0
    for i in dt_list:
        n += 1
        if i <= min_num:
            min_num = i
            num = n
    return num,min_num


def reset_cen(pdict, cdict):
    for key in pdict.keys():
        sum_x = 0
        sum_y = 0
        try :
            for p_x, p_y in pdict[key]:
                sum_x += p_x
                sum_y += p_y
        except BaseException:
            continue
        cen_x = sum_x / pdict[key].shape[0]
        cen_y = sum_y / pdict[key].shape[0]
        cen_new = np.array([[cen_x, cen_y]])
        # if (cdict[key] != np.squeeze(cen_new)).all():
        cdict[key] = np.concatenate([cdict[key], cen_new])


def draw(point_dict, cen_dict):
    # 先用不同颜色画散点图
    plt.ion()  # 交互模式
    fig, ax = plt.subplots()
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    color_n = 0
    for key in point_dict.keys():
        if point_dict[key].all():
            ax.plot(point_dict[key][:, 0], point_dict[key][:, 1], 'ro', color= color[color_n])  # 画随机的散点
            color_n += 1
    ax.set_title('K-means')
    color_n = 2
    plt.pause(0.1)
    # 再画出中心点移动的轨迹
    star = mpath.Path.unit_regular_star(6)
    circle = mpath.Path.unit_circle()
    # concatenate the circle with an internal cutout of the star
    verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
    codes = np.concatenate([circle.codes, star.codes])
    cut_star = mpath.Path(verts, codes)

    for key in cen_dict.keys():
        try:
            ax.plot(cen_dict[key][:, 0], cen_dict[key][:, 1], '--r', marker=cut_star, markersize=10, color=color[color_n])  # 画两个中心点轨迹
        except BaseException:
            print(cen_dict)
            
        color_n += 1
    plt.pause(0.1)


def shoulian(cen_dict):
    num = 0
    for cen in cen_dict.keys():
        num += 1
        if cen_dict[cen].shape[0] > 1:
            if cen_dict[cen][-1][0] == cen_dict[cen][-2][0]:
                num -= 1
    return num == 0


def k_means(point, n, draw_plot = True):
    cen_dict = {}                          # 先定好中心
    for i in range(n):
        name = 'center_' + str(i+1)                # 先随机吧
        min_x = int(min(point[:, 0]) *1.2)
        max_x = int(max(point[:, 0]) *0.8)
        min_y = int(min(point[:, 1]) *1.2)
        max_y = int(max(point[:, 1]) *0.8)
        cen_dict[name] = np.array([[random.randint(min_x, max_x), random.randint(min_y, max_y)]])

    for i in range(100):
        if shoulian(cen_dict):
            break
        point_dict = apart(point, cen_dict)  # 按距离分类
        reset_cen(point_dict, cen_dict)  # 重新定中心
    if draw_plot:
        draw(point_dict,cen_dict)
    return point_dict,cen_dict

def init_center(point,n):
    min(point[:,0])


def make_points_around(x,y,r,n):     #坐标，相对的距离，要生成点集的数量
    p_x = np.random.randint(x-r,x+r,size=n).reshape(n,1)
    p_y = np.random.randint(y-r,y+r,size=n).reshape(n,1)
    return np.hstack((p_x,p_y))
   

def assess_sse(point_dict, cen_dict):           #评价聚类的好坏，sse值越低说明聚类效果越好4
    SSE = 0
    for key in cen_dict.keys():
        cen = cen_dict[key][-1]
        try:
            for i in point_dict[key]:
                SSE += distance_sq(cen, i)
        except BaseException:
            continue
    return SSE

def best_kmean(points, max_k):                #测试并加以评价得出最小而且最优的k值，并返回此条件下的点集,max_k是人工估计的k最大值
    d_value = []                      #sse值的一阶差分
    dd_value = []                     #sse值的二阶差分
    p_d,c_d = k_means(points, 1, draw_plot=False) 
    form_value = assess_sse(p_d, c_d)  
    for i in range(2,max_k+1):
        p_d,c_d = k_means(points, i, draw_plot=False) 
        sse_value = assess_sse(p_d, c_d)  
        d_value.append(abs(sse_value-form_value))
        form_value = sse_value  

    k=1
    form_dvalue = d_value[0]
    for i in range(1,len(d_value)):
        sse_dvalue = abs(d_value[i]-d_value[i-1])
        dd_value.append(sse_dvalue)
        if sse_dvalue>form_dvalue*0.7:
            k = i+1
            break
        form_dvalue = sse_dvalue
    print(d_value)
    print(dd_value)
    print(k)
    return k_means(points, k, draw_plot=True)
    
    

    


# p1 = make_points_around(100,100,20,40)
# p2 = make_points_around(100,200,20,20)
# # p3 = make_points_around(200,200,20,20)
# # p4 = make_points_around(200,100,20,20)
# # 第一步，产生两类随机数据。数据内容有：x,y坐标值
# # num = 20   # 产生每类20个随机数
# # ran = 30   # 随机数在在中心30的距离

# # p1 = nr.normal(100, ran, size=(2, num)).T   #生成在某个数周围的点集
# # p2 = nr.normal(200, ran, size=(2, num)).T
# # p3 = nr.normal(200, ran, size=(2, num)).T
# # print(p1)
# # np.set_printoptions(precision=4)    # 显示保留四位有效数字
# point1 = np.concatenate([p1,p2])   # 合并数据
# # print(point1)
# # p_d,c_d = k_means(point1,5)
# # print(p_d)
# # print(c_d)

# # sse_3 = assess_sse(p_d, c_d)
# # print(sse_3)

# best_kmean(point1, 6)
# plt.pause(15)