import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import math_fun as mf
import random


def shrink(img, times):
    H, W = img.shape
    lH, lW = int(H/times), int(W/times)
    print(lH, lW)
    little_img = np.zeros((lH, lW), dtype=np.uint8)
    for i in range(lH):
        for j in range(lW):
          #  print(i*times, j*times)
            little_img[i][j] = img[i*times][j*times]
    return little_img

def threshold_two(img, threshold):
    shape =img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):  
            if img[i][j] <= threshold:
                img[i][j] = 0
            else:
                img[i][j] = 255





def graychance(inimg, level):     #灰度变换
    shape = inimg.shape

    for i in range(shape[0]):
        for j in range(shape[1]):
            value = inimg[i][j] 
            value = level[0]*value + level[1]
            if value < 0:
                value = 0
            elif value > 255:
                value = 255    
            inimg[i][j] = value    
    print(inimg[0][0])

def zhifang(img):          #直方图均衡化
    tran = np.zeros(256)
    for element in img.flat:
        tran[element] += 1
    tran[0] = tran[0]/img.size*255    
    for n in range(1,255):
        tran[n] = tran[n-1] + tran[n]/img.size*255    
            
    for i in range(img.shape[0]):
        for j in range(img.shape[1]): 
            img[i][j] = tran[img[i][j]]
  #  plt.bar(np.arange(256), tran)
  #  plt.show()   




def numpy_conv(inputs,filter):    #矩阵与卷积核的卷积运算
    H, W = inputs.shape
    filter_size = filter.shape[0]

    result = np.zeros((inputs.shape))       #这里先定义一个和输入一样的大空间，但是周围一圈后面会截掉
    for r in range(0, H - filter_size + 1): #卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置
        for c in range(0, W - filter_size + 1):   
            cur_input = inputs[r:r + filter_size, c:c + filter_size]            
         
            cur_output = cur_input * filter       #和核进行乘法计算
            conv_sum = np.sum(cur_output) #再把所有值求和
     
            result[r, c] = int(conv_sum)
    return result.astype(np.uint8)


def mid_value_filter(inputs,size):     #中值滤波
    H, W = inputs.shape
    result = np.zeros((inputs.shape),dtype=int)       #这里先定义一个和输入一样的大空间

    for r in range(0, H - size + 1):        #卷积核通过输入的每块区域
        for c in range(0, W - size + 1):      
            cur_list = inputs[r:r + size, c:c + size].ravel()
            mid = get_mid(cur_list)
            result[r, c] = mid
    return result.astype(np.uint8)


def get_merge(inputs, filter, threshold):
    H, W = inputs.shape
    filter_size = filter.shape[0]

    result = np.zeros((inputs.shape))       #这里先定义一个和输入一样的大空间，但是周围一圈后面会截掉
    for r in range(0, H - filter_size + 1): #卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置
        for c in range(0, W - filter_size + 1):   
            cur_input = inputs[r:r + filter_size, c:c + filter_size]            
         
            cur_output = cur_input * filter       #和核进行乘法计算
            conv_sum = np.sum(cur_output) #再把所有值求和
            if conv_sum <= threshold:
                conv_sum = 0
            result[r, c] = int(conv_sum)
    return result.astype(np.uint8)

def get_mid(array):
  
    size = array.size
    array = np.sort(array)
    midnum = int(size/2)
    if size % 2 == 0:
        mid = int((array[midnum-1] + array[midnum])/2)
    elif size % 2 ==1:
        mid = array[midnum]  
    return mid


def filter_aver(size):    #邻域滤波的卷积核
    return np.ones((size,size)) /(size**2)


def find_marea(img):
    H, W = img.shape
    sum_x = np.zeros(W, dtype=np.int16)
    sum_y = np.zeros(H, dtype=np.int16)
    print(H, W)
    for i in range(W):
        sum_x[i] = img[:,i].sum()

    for j in range(H):
        sum_y[j] = img[j,:].sum()
    print(sum_x)
    print(sum_y)

def ran_hough(img):
    #第一步  将二值图中不为0的点的坐标存入一个数组中
    #第二步  从数组中随机选取三个点，得到三个点连成线段的（中点），
    #       得出三个点的（切线方程） ，由方程得到两个（交点），与相应的直线
    #       方程上的中点（连成两条直线），再得到其（交点）作为中心
    #第三步  中心以及三个点待入椭圆方程，a(x − p)2 + 2b(x − p)(y − q) + c(y − q)2 + 1 = 0 中，（解得椭圆）
    #第四步  用一个列表保存此椭圆参数，再次进行第二步，得到新椭圆与就椭圆（比较），
    #        若相似度差太多，则把新椭圆加入数组中，若差不多，则取两椭圆平均，并把相应权值加1
    #第五步 
    all_points = get_points(img)
    num_p = all_points.shape[0]
    print(num_p)

    p1 = all_points[random.randint(0， num_p-1)]       #获取三个随机点
    p2 = all_points[random.randint(0， num_p-1)]
    p3 = all_points[random.randint(0， num_p-1)]
    
    m_p1p2 = mf.mid_point(p1, p2)                     #计算三点的中点
    m_p2p3 = mf.mid_point(p2, p3)
    m_p1p3 = mf.mid_point(p1, p3)

    img_around_p1 = get_area_around(img, p1, 5)          #获取三点附近区域
    img_around_p2 = get_area_around(img, p1, 5)
    img_around_p3 = get_area_around(img, p1, 5)

    p_around_p1 = get_points(img_around_p1)               #获取三点附近区域的点
    p_around_p2 = get_points(img_around_p2)
    p_around_p3 = get_points(img_around_p3)

    line_k1 ,line_b1 = mf.OLS(p_around_p1)                #分别用三点附近区域的点回归出三条直线的参数
    line_k2 ,line_b2 = mf.OLS(p_around_p2)
    line_k3 ,line_b3 = mf.OLS(p_around_p3)
    
    cline_1 = mf.Line(line_k1,line_b1)                  #用参数获得三条切线直线
    cline_2 = mf.Line(line_k2,line_b2)
    cline_3 = mf.Line(line_k3,line_b3)

    cross_p1p2 = cline_1.cross_point(cline_2)            #获得三条直线相交的两个交点
    cross_p2p3 = cline_2.cross_point(cline_3)

    mline_1 = mf.Line(p1=(c))


    cline_p1 = mf.Line(p1=m_p1p2, p2=cross_p1p2)
    cline_p2 = mf.Line(p1=m_p2p3, p2=cross_p2p3)

    pass


def get_points(img):                #得到二值图里不为零的点
    H, W = img.shape
    p_list = []
    for i in range(H):
        for j in range(W):
            if img[i][j] != 0:
                p_list.append([i,j])
    return np.array(p_list)

def get_area_around(img, point, size):   #得到图像中某点周围的一部分
    H, W = img.shape

    if point[1] - size < 0:
        left = 0
    else:
        left = point[1] - size
    if point[1] + size > W:
        right = W-1
    else:
        right = point[1] + size

    if point[0] - size < 0:
        up = 0
    else:
        up = point[0] - size
    if point[1] + size > H:
        down = H-1
    else:
        down = point[0] + size
    
    print(up,down,left,right)
    return img[up:down,left:right]




def print_zhifan(img):
    matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
    plt.hist(img.ravel(), bins=40, normed=0, facecolor="blue",
         edgecolor="black", alpha=0.7)    # 显示横轴标签
    plt.xlabel("区间")    # 显示纵轴标签
    plt.ylabel("频数/频率")    # 显示图标题
    plt.title("频数/频率分布直方图")
    plt.show()


def print_inf_of_img(img):  
    print("数据类型",type(img))           #打印数组数据类型  
    print("数组元素数据类型：",img.dtype) #打印数组元素数据类型  
    print("数组元素总数：",img.size)      #打印数组尺寸，即数组元素总数  
    print("数组形状：",img.shape)         #打印数组形状  
    print("数组的维度数目",img.ndim)  
    print(img[0,0])

def show(img, strs):
    cv2.imshow(strs, img)
    cv2.waitKey(0)
   
robot_filter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
one_filter = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])

img1 = cv2.imread('A.jpg', 0)
img1 = shrink(img1, 2)
print_inf_of_img(img1)

zhifang(img1)
show(img1,"imgzhifan")




#graychance(img1,(1.1,-10))
#show(img1,"imggray")

#img2 = numpy_conv(img1,filter_aver(5))
#show(img2, "imgaver")

# img3 = mid_value_filter(img1, 5)
# show(img3, "imgmid")

# img4 = get_merge(img3, robot_filter, 40)
# show(img4, "imgmerge")

# threshold_two(img4, 40)
# show(img4, "imgthreshold")


cv2.destroyAllWindows()





