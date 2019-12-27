import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import math_fun as mf
import random


def shrink(img, times):                  #收缩图像
    H, W = img.shape
    lH, lW = int(H/times), int(W/times)
    little_img = np.zeros((lH, lW), dtype=np.uint8)
    for i in range(lH):
        for j in range(lW):
          #  print(i*times, j*times)
            little_img[i][j] = img[i*times][j*times]
    return little_img


def threshold_two(img, threshold):       #将图像转化为二值图
    shape =img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):  
            if img[i][j] <= threshold:
                img[i][j] = 0
            else:
                img[i][j] = 255


def clean_along_points(img, size):      #清除二值图中那些孤立的点
    H, W = img.shape
    for i in range(size, H - size + 1, 2*size+1):
        for j in range(size, W - size + 1, 2*size+1):
            points = get_points_around(img, (j, i), size)
            if points.shape[0] <= int(size**2*0.1):
                img[i - size : i + size, j - size : j + size] = 0

    
def flood_fill(img, x, y, l=3):         #使用循环的洪泛填充法对提取出的图像边框进行填充
    H, W = img.shape
    stack = [(x,y)]  
    while any(stack):
        flag = False
        x, y = stack.pop()
        if x <l or x> W-l-1 or y<l or y>H-l-1:
            break

        if img[y - 1, x - 1] == 200 and img[y -1, x] == 200 and img[y - 1, x + 1] == 200\
            and img[y, x -1] == 200 and img[y, x + 1] == 200 and img[y + 1, x - 1] == 200\
                and img[y + 1, x] == 200 and img[y + 1, x + 1] == 200:      
                    continue

        for i in range(-l, l+1):
            for j in range(-l, l+1):
                if img[y + i, x + j] == 255:
                    flag = True 
        if flag:
            continue

        for i in range(-1,2):
            for j in range(-1, 2): 
                if img[y + i, x + j] != 200:        #若未访问，则涂色入栈  
                    img[y + i, x + j] = 200
                    stack.append((x + j, y + i))
                

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


def zhifang(img):                 #直方图均衡化
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


def numpy_conv(inputs,filter):                  #矩阵与卷积核的卷积运算
    H, W = inputs.shape
    filter_size = filter.shape[0]

    result = np.zeros((inputs.shape))           #这里先定义一个和输入一样的大空间，但是周围一圈后面会截掉
    for i in range(0, H - filter_size + 1):     #卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置
        for j in range(0, W - filter_size + 1):   
            cur_input = inputs[i:i + filter_size, j:j + filter_size]            
         
            cur_output = cur_input * filter     #和核进行乘法计算
            conv_sum = np.sum(cur_output)       #再把所有值求和
     
            result[i, j] = int(conv_sum)
    return result.astype(np.uint8)


def mid_value_filter(inputs,size):               #中值滤波
    H, W = inputs.shape
    result = np.zeros((H-size+1, W-size+1),dtype = np.uint8)     #这里先定义一个和输入一样的大空间
 
    for r in range(0, H - size + 1):             #卷积核通过输入的每块区域
        for c in range(0, W - size + 1):      
            cur_list = inputs[r:r + size, c:c + size].ravel()
            mid = get_mid(cur_list)
            result[r, c] = mid
    return result.astype(np.uint8)


def get_merge(inputs, filter, threshold):         #提取边缘，可采用不同算子
    H, W = inputs.shape
    filter_size = filter.shape[0]

    result = np.zeros((inputs.shape))             #这里先定义一个和输入一样的大空间，但是周围一圈后面会截掉
    for r in range(0, H - filter_size + 1):       #卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置
        for c in range(0, W - filter_size + 1):   
            cur_input = inputs[r:r + filter_size, c:c + filter_size]            
         
            cur_output = cur_input * filter       #和核进行乘法计算
            conv_sum = np.sum(cur_output)         #再把所有值求和
            if conv_sum <= threshold:
                conv_sum = 0
            result[r, c] = int(conv_sum)
    return result.astype(np.uint8)


def get_mid(array):          #求中位数
  
    size = array.size
    array = np.sort(array)
    midnum = int(size/2)
    if size % 2 == 0:
        mid = int((array[midnum-1] + array[midnum])/2)
    elif size % 2 ==1:
        mid = array[midnum]  
    return mid


def filter_aver(size):       #生成邻域滤波的卷积核
    return np.ones((size,size)) /(size**2)


def ran_hough(img, evident):                               #随机霍夫变换拟合椭圆
    #第一步  将二值图中不为0的点的坐标存入一个数组中
    #第二步  从数组中随机选取三个点，得到三个点连成线段的（中点），
    #       得出三个点的（切线方程） ，由方程得到两个（交点），与相应的直线
    #       方程上的中点（连成两条直线），再得到其（交点）作为中心
    #第三步  中心以及三个点待入椭圆方程，a(x − p)2 + 2b(x − p)(y − q) + c(y − q)2 + 1 = 0 中，（解得椭圆）
    #第四步  用一个列表保存此椭圆参数，再次进行第二三步，得到新椭圆与就椭圆（比较），
    #        若（相似度）差太多，则把新椭圆加入数组中，若差不多，则取两椭圆平均，并把相应权值加1
    #第五步  输出权值大于门槛的椭圆对象
    all_points = get_points(img)
    num_p = all_points.shape[0]
    ovals = []
    print(num_p)
    
    for i in range(2000):       
        p1 = all_points[random.randint(0, num_p-1)]         #获取三个随机点
        p2 = all_points[random.randint(0, num_p-1)]
        p3 = all_points[random.randint(0, num_p-1)]
        print("p1:"+str(p1)+", p2:"+str(p2)+", p3:"+str(p3))
        m_p1p2 = mf.mid_point(p1, p2)                       #计算三点的中点
        m_p2p3 = mf.mid_point(p2, p3)
        m_p1p3 = mf.mid_point(p1, p3)
        print("m_p1p2:"+str(m_p1p2)+", m_p2p3:"+str(m_p2p3)+", m_p1p3:"+str(m_p1p3))
        
        p_around_p1 = get_points_around(img, p1, 8)         #获取三点附近区域的点
        p_around_p2 = get_points_around(img, p2, 8)
        p_around_p3 = get_points_around(img, p3, 8)

        line_k1 ,line_b1 = mf.OLS(p_around_p1)              #分别用三点附近区域的点回归出三条直线的参数
        line_k2 ,line_b2 = mf.OLS(p_around_p2)
        line_k3 ,line_b3 = mf.OLS(p_around_p3)
        
        cline_1 = mf.Line(line_k1,line_b1)                  #用参数获得三条切线直线
        cline_2 = mf.Line(line_k2,line_b2)
        cline_3 = mf.Line(line_k3,line_b3)

        cross_p1p2 = cline_1.cross_point(cline_2)           #获得三条直线相交的两个交点
        cross_p2p3 = cline_2.cross_point(cline_3)
        if not (cross_p1p2 and cross_p2p3):
            continue

        cline_p1 = mf.Line(p1=m_p1p2, p2=cross_p1p2)        #切线的两个交点于对应两个中点连成线
        cline_p2 = mf.Line(p1=m_p2p3, p2=cross_p2p3)
        
        cen = cline_p1.cross_point(cline_p2)                #得到椭圆的中心点
        print("center is： "+str(cen))
        if not cen:
            continue

        oval_new = mf.Oval(cen=cen, p1=p1, p2=p2, p3=p3)
        if not oval_new.enable:
            continue 
        oval_new.print_fomula()

        len_ovals = len(ovals)
        flag_append = True
        if len_ovals == 0:
            ovals.append(oval_new)
        else:
            for i in range(len_ovals):                    #将新的椭圆于已有椭圆相比较，若相似则融合，若不相似则加入列表
                if ovals[i].similar(oval_new, 0.75):      #九成相似即为相似
                    flag_append = False
                    ovals[i].fuse(oval_new)                                 
                    if ovals[i].evident > evident:        #若列表中有超过权值的椭圆，返回此椭圆
                        return ovals[i]     
                #print("evident: " + str(ovals[i].evident)) 
            if flag_append:    
                ovals.append(oval_new)       
                
    maxs = 0    
    max_i = 0
    for i in range(len(ovals)):
        if ovals[i].evident >= maxs:
            maxs = ovals[i].evident
            max_i = i
    return ovals[max_i]


def get_points(img):                      #得到二值图里不为零的点
    H, W = img.shape
    p_list = []
    for i in range(H):
        for j in range(W):
            if img[i][j] != 0:
                p_list.append([j,i])
    #print("points in all area"+ str(p_list))
    return np.array(p_list)


def get_points_around(img, point, size):   #得到图像中某点周围的非零点
    H, W = img.shape
    p_list = []

    if point[0] - size <= 0:
        left = 0
    else:
        left = point[0] - size
    if point[0] + size > W:
        right = W
    else:
        right = point[0] + size

    if point[1] - size <= 0:
        up = 0
    else:
        up = point[1] - size
    if point[1] + size > H:
        down = H
    else:
        down = point[1] + size
    
    for i in range(up, down):
        for j in range(left, right):
            if img[i][j] != 0:
                p_list.append([j,i]) 
    
  #  print(up,down,left,right)
   # print("points in area"+ str(p_list))
    return np.array(p_list)
  

def print_zhifan(img):
    matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
    plt.hist(img.ravel(), bins=40, normed=0, facecolor="blue",
         edgecolor="black", alpha=0.7)                  # 显示横轴标签
    plt.xlabel("区间")                 # 显示纵轴标签
    plt.ylabel("频数/频率")             # 显示图标题
    plt.title("频数/频率分布直方图")
    plt.show()


def print_inf_of_img(img):  
    print("数据类型",type(img))            #打印数组数据类型  
    print("数组元素数据类型：",img.dtype)   #打印数组元素数据类型  
    print("数组元素总数：",img.size)        #打印数组尺寸，即数组元素总数  
    print("数组形状：",img.shape)           #打印数组形状  
    print("数组的维度数目",img.ndim)  
    print(img[0,0])

def show(img, strs):
    cv2.imshow(strs, img)
    cv2.waitKey(0)
   

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(str(int(x))+" , "+str(int(y)))
        cv2.imshow("image", img4)

robot_filter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])    #罗伯特算子
one_filter = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])    #垂直梯度算子

img1 = cv2.imread('A.jpg', 0)    #读入图像

img1 = shrink(img1, 2)           #图像收缩
show(img1,"img")

zhifang(img1)                    #直方图均衡化
show(img1,"imgzhifan")

#graychance(img1,(1.1,-10))      #灰度变换
#show(img1,"imggray")

#img2 = numpy_conv(img1,filter_aver(5))     #均值滤波
#show(img2, "imgaver")

img3 = mid_value_filter(img1, 5)            #中值滤波
show(img3, "imgmid")

img4 = get_merge(img3, robot_filter, 10)    #提取边缘
show(img4, "imgmerge")

print_inf_of_img(img4)        
threshold_two(img4, 50)                 #转换二值图
show(img4, "image_two")

clean_along_points(img4, 8)             #清除孤立杂点
show(img4, "image_clean")
 
oval1 = ran_hough(img4, 8)              #随机霍夫变换
oval1.print_fomula()
print(oval1.angle,oval1.lone_axis,oval1.short_axis)
print(oval1.check_point([504,508]))     #越小说明点距离椭圆越近
print(oval1.check_point([498,190]))

#flood_fill(img4, int(img4.shape[1]/2), int(img4.shape[0]/2))     #洪泛填充
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img4)
cv2.waitKey(0)






