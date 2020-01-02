import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import math_fun as mf
import random
import math
import time
from kmeans import best_kmean

######################基本预处理函数####################
def shrink(img, times=0, mianji=320000):         #收缩图像，mainji为最后要收缩到的总像素数
    H, W =img.shape[0:2] 
    if times == 0:
        times = math.sqrt(H*W / mianji)
    lH, lW = int(H/times), int(W/times)
    
    if len(img.shape) <=2:
        little_img = np.zeros((lH, lW), dtype=np.uint8)
    else:  
        little_img = np.zeros((lH, lW, 3), dtype=np.uint8)

    for i in range(lH):
        for j in range(lW):
          #  print(i*times, j*times)
            little_img[i,j] = img[int(i*times),int(j*times)]
    return little_img

def cut_image(img,point_lu,point_rd):    #切割图像，并把切割出来的部分复制出来返回，参数为左上和右下两个点的坐标
    return img[point_lu[1]: point_rd[1]+1, point_lu[0]:point_rd[0]]
    


def rgb_turn_gray(img):                 #转换为灰度图
    H, W = img.shape[0:2]
    for i in range(H):
        for j in range(W):
            gray = (img[i,j][0]*38 + img[i,j][1]*75 + img[i,j][2]*15) >> 7
            img[i,j] = gray


def threshold_two(img, threshold):       #将图像转化为二值图，threshold为阈值
    shape =img.shape
    for i in range(shape[0]):
        for j in range(shape[1]):  
            if img[i][j] <= threshold:
                img[i][j] = 0
            else:
                img[i][j] = 255
       

def graychance(inimg, level):     #灰度变换，leve为元组或列表，由斜率和截距组成
    shape = inimg.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            value = inimg[i][j]
            value = level[0]*value + level[1]
            if value < 1:
                value = 0
            elif value > 254:
                value = 255    
            inimg[i][j] = value    
    

def zhifang(img):                 #直方图均衡化
    H, W = img.shape
    tran = np.zeros(256)
    for element in img.flat:
        tran[element] += 1

    tran[0] = tran[0]/img.size*255    
    for n in range(1,255):
        tran[n] = tran[n-1] + tran[n]/img.size*255         
    for i in range(H):
        for j in range(W): 
            img[i,j] = tran[img[i,j]]


        
    # plt.bar(np.arange(256), tran)
    # plt.show()   


def rgb_turn_hsi(rgb):       #rgb值转换为hsi值
    # print(rgb)
    r = rgb[0]/255
    g = rgb[1]/255
    b = rgb[2]/255
    i = (r+g+b)/3
    if any(rgb):
        s = 1 - min(r,g,b)*3/(r+g+b)
    else:
        s = 0
    in_sqrt = r*r+g*g+b*b-r*g-b*g-b*r
    if in_sqrt<=0:
        sita = 0
    else:
        in_acos = (r-g/2-b/2)/math.sqrt(in_sqrt)
        if abs(in_acos)>1:
            sita = 0
        else:
            sita = math.acos(in_acos)
    sita = sita*180/math.pi
    if g>=b:
        h = sita
    else:
        h = 360-sita
    return [h,s,i]


def rgb_turn_hsi_img(img):     #rgb图转换为hsi图
    H, W = img.shape[0:2]
    for i in range(H):
        for j in range(W):
            hsi = rgb_turn_hsi(img[i,j])
            hsi[0] = hsi[0] /360*255
            hsi[1] = hsi[1]*255
            hsi[2] = hsi[2]*255        
            img[i,j] = np.array(hsi,dtype=np.uint8)


def channal_div(img):    #rgb图像拆分为r,g,b通道图
    H, W = img.shape[0:2]
    b, g, r =np.dsplit(img,3)
    return [b.reshape(H,W),g.reshape(H,W),r.reshape(H,W)]

def channal_com(img_r, img_g, img_b):       #r,g,b通道图合并为rgb图
    return np.stack([img_b, img_g, img_r], axis=2)

#########################卷积图像操作函数########################
robot_filter = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])    #罗伯特算子
one_filter = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])    #垂直梯度算子

def aver_filter(size):       #生成邻域滤波的卷积核
    return np.ones((size,size)) /(size**2)


def gauss_filter(size,sigema = 0):              #生成高斯卷积核
    if not sigema:
        sigema =size/6-1/6
        # sigema2 = 0.3*((size-1)*0.5-1)+0.8
    core = np.zeros((size,size), dtype=np.float)
    r = int(size/2)
    for x in range(-r,r+1):
        for y in range(-r,r+1):
            G = (1/(2*math.pi*sigema**2))*math.exp(-(x**2+y**2)/(2*sigema**2))
            core[r-y,r+x] = G
    core = core/np.sum(core)
   # print(str(core))
    return core
    

def numpy_conv(inputs,filter):                  #矩阵与卷积核的卷积运算
    H, W = inputs.shape
    filter_size = filter.shape[0]

    result = np.zeros((H-filter_size+1, W-filter_size+1),dtype = np.uint8)     #这里先定义一个和输入一样的大空间
    for i in range(0, H - filter_size + 1):     #卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置
        for j in range(0, W - filter_size + 1):   
            cur_input = inputs[i:i + filter_size, j:j + filter_size]            
         
            cur_output = cur_input * filter     #和核进行乘法计算
            conv_sum = np.sum(cur_output)       #再把所有值求和
     
            result[i, j] = conv_sum
    return result.astype(np.uint8)


def mid_value_filter(inputs,size):               #中值滤波
    H, W = inputs.shape
    result = np.zeros((H-size+1, W-size+1),dtype = np.uint8)     #这里先定义一个和输入一样的大空间
 
    for r in range(0, H - size + 1):             #卷积核通过输入的每块区域
        for c in range(0, W - size + 1):      
            cur_list = inputs[r:r + size, c:c + size].ravel()
            mid = mf.get_mid(cur_list)
            result[r, c] = mid
    return result.astype(np.uint8)


def Masaike_filter(inputs, size, color, threshold=0.5):               #马赛克滤波
    H, W = inputs.shape[0:2]
    l_color = np.array([[color[2],color[1],color[0]]])
    s_color = np.squeeze(l_color)
    white = np.array([255,255,255])
    
    s_points = []
    white_points = []
    sq =size**2
    for i in range(0, H - size + 1, size):             #卷积核通过输入的每块区域
        for j in range(0, W - size + 1, size):  
            area = inputs[i:i + size, j:j + size].reshape(sq,1,3)
            s = area[:,0]==l_color 
            sum_p = np.sum(s[:,0]*s[:,1]*s[:,2])
            if sum_p > threshold*sq:             
                inputs[i:i + size, j:j + size] = s_color
                s_points.append([int(j+size/2),int(i+size/2)])
            else:
                inputs[i:i + size, j:j + size] = white
                white_points.append([int(j+size/2),int(i+size/2)])
                    
    return np.array(s_points),np.array(white_points)        


####################后续增强特征和消除噪声的辅助函数####################
def clean_along_points(img, size):      #清除二值图中那些孤立的点，size表范围
    H, W = img.shape
    all_points = get_points(img)
    for point in all_points:
        points = get_points_around(img, (point[0], point[1]), size)
        if points.shape[0] <= int(size**2*0.11):
            #img[i - size : i + size, j - size : j + size] = 0
            img[point[1], point[0]] = 0


def get_merge(inputs, filter, threshold_low=5,threshold_high=254):         #提取边缘，可采用不同算子
    H, W = inputs.shape
    filter_size = filter.shape[0]
    result = np.zeros((inputs.shape))             #这里先定义一个和输入一样的大空间，但是周围一圈后面会截掉
    for r in range(0, H - filter_size + 1):       #卷积核通过输入的每块区域，stride=1，注意输出坐标起始位置
        for c in range(0, W - filter_size + 1):   
            cur_input = inputs[r:r + filter_size, c:c + filter_size]            
         
            cur_output = cur_input * filter       #和核进行乘法计算
            conv_sum = abs(np.sum(cur_output))         #再把所有值求和
            if conv_sum <= threshold_low:
                conv_sum = 0
            elif conv_sum >= threshold_high:
                conv_sum = 255
            result[r, c] = conv_sum
    return result.astype(np.uint8)


#增强初步提取出的图像边缘的信息，把与强边缘连通的弱边缘增强
def edge_follow(img, points,threshold_low=100, threshold_high=200):   
    H = points.shape[0]
    print(H)
    flags = np.zeros((H,1),dtype=np.int32)               #在点集中设置标志位
    points = np.hstack((points, flags))
    stack = []
    queue = []
    connected = False
    order = np.arange(H)
    points = np.hstack((points, order.reshape(H,1)))
    for point in points:
        if point[2]==0 and img[point[1],point[0]] < threshold_low:
            point[2] == 1
            stack.append(point)
            queue.append(point)
            while np.any(stack):
                point2 = stack.pop()
                x = point2[0]
                y = point2[1]
                for i in range(-1,2):
                    for j in range(-1,2):
                        if i==0 and j==0:
                            continue
                        if img[y+j,x+i] and img[y+j,x+i] < threshold_low:
                            b1 = points[:,0] == x+i                 #使用numpy花式索引技巧，不用遍历查找点
                            b2 = points[:,1] == y+j                        
                            p = order[b1 * b2][0]        #速度是遍历的两百倍左右
                            if points[p][2] == 0:
                                points[p][2] = 1
                                stack.append(points[p])
                                queue.append(points[p])
                        elif img[y+j, x+i] > threshold_high:
                            connected = True
            if connected == False:
                while len(queue) >0:
                    q_point = queue.pop(0)
                    points[q_point[3]][2] = -1
            else:
                queue = []    
                connected = False   
    for point in points:
        if point[2]==1:
            img[point[1], point[0]] = 255
                            

#########################形状识别函数##########################
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
    j=0
    for i in range(400000):   
        if len(ovals) > 0 and len(ovals) > 20000:
            j=0
            while j <= (len(ovals)-1):
                if ovals[j].evident == 1:
                    del ovals[j]
                    j = j-1
                j = j + 1
        p1 = all_points[random.randint(0, num_p-1)]         #获取三个随机点
        p2 = all_points[random.randint(0, num_p-1)]
        p3 = all_points[random.randint(0, num_p-1)]
       # print("p1:"+str(p1)+", p2:"+str(p2)+", p3:"+str(p3))
        m_p1p2 = mf.mid_point(p1, p2)                       #计算三点的中点
        m_p2p3 = mf.mid_point(p2, p3)
        m_p1p3 = mf.mid_point(p1, p3)
        #print("m_p1p2:"+str(m_p1p2)+", m_p2p3:"+str(m_p2p3)+", m_p1p3:"+str(m_p1p3))
        
        p_around_p1 = get_points_around(img, p1, 6)         #获取三点附近区域的点
        p_around_p2 = get_points_around(img, p2, 6)
        p_around_p3 = get_points_around(img, p3, 6)

        cline_1 = mf.OLS(p_around_p1)              #分别用三点附近区域的点回归出三条直线的参数
        cline_2 = mf.OLS(p_around_p2)
        cline_3 = mf.OLS(p_around_p3)
        
        # img0 = np.copy(img)
        # delete_linepoints(img0, cline_1, gray=255, size=2)
        # img0[p1[1]-3:p1[1]+4,p1[0]-3:p1[0]+4]=150
        # show(img0, "0000", live_time=0)

        cross_p1p2 = cline_1.cross_point(cline_2)           #获得三条直线相交的两个交点
        cross_p2p3 = cline_2.cross_point(cline_3)
        if not (cross_p1p2 and cross_p2p3):
            continue

        cline_p1 = mf.Line2(p1=m_p1p2, p2=cross_p1p2)        #切线的两个交点于对应两个中点连成线
        cline_p2 = mf.Line2(p1=m_p2p3, p2=cross_p2p3)
        
        cen = cline_p1.cross_point(cline_p2)                #得到椭圆的中心点
        # print("center is： "+str(cen))
        if not cen:
            continue
        oval_new = mf.Oval(cen=cen, p1=p1, p2=p2, p3=p3)
        if not oval_new.enable:
            continue 
        # oval_new.print_fomula()
        # print("what?")
        # img15 = make_image(oval_new.points_on_oval())   #根据拟合出的椭圆画图
        # show(img15, "end")
        len_ovals = len(ovals)
        flag_append = True
        if len_ovals == 0:
            ovals.append(oval_new)
        else:
            for i in range(len_ovals):                    #将新的椭圆于已有椭圆相比较，若相似则融合，若不相似则加入列表
                if ovals[i].similar(oval_new, 0.8):      #九成相似即为相似
                    flag_append = False
                    ovals[i].fuse(oval_new)                                 
                    if ovals[i].evident > evident:        #若列表中有超过权值的椭圆，返回此椭圆
                        print("time"+str(i))
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
    print("time     "+str(i))
    return ovals[max_i]


def Hough_line(img, evident,lines_num):
    all_points = get_points(img)
    num_p = all_points.shape[0]
    lines = []
    out_lines = []
    print(num_p)
    img_cp = np.copy(img)

    
    for i in range(200000):
        if len(lines) > 0 and len(lines)%10000 == 0:
            j=0
            while j < (len(lines)-1):
                if lines[j].evident == 1:
                    del lines[j]
                    j = j-1
                j = j + 1
        p1 = all_points[random.randint(0, num_p-1)]         #获取1个随机点
        p_around_p1 = get_points_around(img, p1, 11)         #获取三点附近区域的点
        line_new = mf.OLS(p_around_p1)              #分别用三点附近区域的点回归出三条直线的参数

        # img0 = np.copy(img)
        # delete_linepoints(img0, line_new, gray=255, size=2)
        # img0[p1[1]-3:p1[1]+4,p1[0]-3:p1[0]+4]=150
        # show(img0, "0000", live_time=0)

        len_lines = len(lines)
        flag_append = True
        if len_lines == 0:
            lines.append(line_new)
        else:
            j=0         
            while j < len(lines):              #将新的椭圆于已有椭圆相比较，若相似则融合，若不相似则加入列表
                if lines[j].similar(line_new, 0.96):      #九成相似即为相似
                    flag_append = False
                    lines[j].fuse(line_new)                                 
                    if lines[j].evident > evident:        #若列表中有超过权值的椭圆，返回此椭圆
                        out_lines.append(lines[j])
                        if len(out_lines) >= lines_num:  
                            return out_lines
                        
                        delete_linepoints(img, lines[j])

                        # delete_linepoints(img_cp, lines[j], 255)
                        # show(img_cp, "end",0)

                        all_points = get_points(img)
                        num_p = all_points.shape[0]
                        del lines[j] 
                        j = j - 1
                j = j + 1                  
                #print("evident: " + str(ovals[i].evident)) 
            if flag_append:    
                lines.append(line_new)     
    
    return out_lines


#######################图中点集处理函数######################
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
  
def get_RGB_points(img, color, distance=75):
    H, W = img.shape[0:2]
    points = []
    for i in range(H):
        for j in range(W):
            if (color[0]-img[i,j,0])**2+(color[1]-img[i,j,1])**2+(color[2]-img[i,j,2])**2<distance:
                points.append([j,i])
    return np.array(points)

def get_HSI_points(img, color, distance=100):
    H, W = img.shape[0:2]
    points = []
    for i in range(H):
        for j in range(W):
            if np.all(img[i,j,1:3]<82) and color[1]<70 and color[2]<82:
                points.append([j,i])
            elif 2*(color[0]-img[i,j,0])**2+(color[1]-img[i,j,1])**2+(color[2]-img[i,j,2])**2<distance:
                points.append([j,i])
    return np.array(points)


def clean_points(img, points, limit):         #清除点集中不满足条件的点，limit是自写的限制函数参数
    j=0
    while j < points.shape[0]:
        if limit(img, points[j]):
            points = np.delete(points, j, axis=0)
            j = j-1
        j = j + 1
    return points

def biankuang(img, point):
    H, W = img.shape[0:2]
    if point[0] < W *0.08 or point[0] > W*0.92:
        return True
    if point[1] < H*0.1 or point[1] > H*0.9:
        return True
    return False


def get_points_img(img, points):          #得到点集覆盖的图中周围区域，并复制出来返回
    x_l = min(points[:,0])
    x_r = max(points[:,0])
    y_u = min(points[:,1])
    y_d = max(points[:,1])
    p_lu = [x_l-30,y_u-30]
    p_rd = [x_r+30,y_d+30]

    return cut_image(img, p_lu, p_rd)
    


#######################绘图与展示函数########################
def delete_linepoints(img, line, gray=0, size=8):      #在图上绘制直线，默认为删除图中直线上的非零点
    H, W = img.shape

    for i in range(W):
        y = line.fun(i)
        y[1] = int(y[1])
        if y[0] == 'x':    
            print("x = something")
            if y[1]>=size and y[1]<=W-size:
                img[:,y[1]-size:y[1]+size] = gray
            elif y[1]<size:
                img[:,0:y[1]+size] = gray
            elif y[1]>W-size:
                img[:,y[1]-size:W] = gray
            break 
        elif y[1]>=size and y[1]<=H-size:
            img[y[1]-size:y[1]+size, i] = gray
        elif y[1]<size and y[1]>=0:
                img[0:y[1]+size,i] = gray
        elif y[1]>W-size and y[1]<W:
                img[y[1]-size:H,i] = gray
           

def draw_point(img, points, color):    #在图上描点，颜色参数输入rgb或者灰度值
    H, W = img.shape[0:2]
    if type(color)==type((1,1)):
        color = np.array([color[2],color[1],color[0]])
    for point in points:
        if point[1]<H and point[0]<W:
            img[point[1], point[0]] = color


def make_image(points):                                #普通的描点画图
    if points.shape[0] < 3:
        return np.zeros((450,450), dtype=np.uint8)
    max_px = points[:, 0].max()
    min_px = points[:, 0].min()
    max_py = points[:, 1].max()
    min_py = points[:, 1].min()
    # print(str(max_px)+','+str(min_px))
    # print(str(max_py)+','+str(min_py))
    if min_px < 0:
        min_px = 0
    if min_py < 0:
        min_py = 0
    if max_px > 1000:
        max_px = 1000
    if max_py > 1000:
        max_py = 1000 
    H = max_py  + 50
    W = max_px  + 50
    # print(str(H)+','+str(W))
    img = np.zeros((H, W), dtype=np.uint8)
    for point in points:
        if point[0] <0 or point[1]<0 or point[0] >1000 or point[1]>1000:
            continue
        img[point[1], point[0]] = 255
    
    return img


def flood_fill(img, x, y, l=3):         #使用循环的洪泛填充法对提取出的图像边框进行填充
    H, W = img.shape                    #为什么不用递归，因为栈爆了
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


def print_zhifan(img):                  #画直方图
    tran = np.zeros(256)
    for element in img.flat:
        if element:
            tran[element] += 1
    plt.bar(np.arange(256), tran)
    plt.show()   


def print_inf_of_img(img):  
    print("数据类型",type(img))            #打印数组数据类型  
    print("数组元素数据类型：",img.dtype)   #打印数组元素数据类型  
    print("数组元素总数：",img.size)        #打印数组尺寸，即数组元素总数  
    print("数组形状：",img.shape)           #打印数组形状  
    print("数组的维度数目",img.ndim)  
    print(img[0,0])

def show(img, strs,live_time=1):       #显示图片，live_time为0为按任意键继续
    cv2.imshow(strs, img)
    cv2.waitKey(live_time)
   

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):   #用于显示坐标的回调函数之中
    if event == cv2.EVENT_LBUTTONDOWN:
        print(str(int(x))+" , "+str(int(y))+' color : '+str(img1[y,x]))
        cv2.imshow("image", img1)

v_filter = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
x_filter = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
#########################测试区域##########################

img1 = cv2.imread('G.jpg', 1)                         #读入图像
img1 = shrink(img1,mianji=240000)                     #缩小
# img_01 = np.copy(img1)                              #备份
show(img1, "origin image",1)


rgb_turn_hsi_img(img1)                                #转hsi图像
tps = get_HSI_points(img1, (40,40,40))                #提取餐托
draw_point(img1, tps, (0,255,0))                      #画餐托
g_ps, w_ps = Masaike_filter(img1,20,(0,255,0),0.2)    #马赛克化
ww_ps = clean_points(img1, w_ps, biankuang)           #清除在边框的点
p_dirt, c_dirt = best_kmean(ww_ps, 5)                 #k值聚类，大概确定每个餐盘位置

 
img01 = cv2.imread('G.jpg', 0)                        #读入图像
img02 = shrink(img01,mianji=240000)                   #图像收缩
zhifang(img02)                                        #直方图均衡化
img03 = mid_value_filter(img02, 7)                    #中值滤波
img04 = get_merge(img03, robot_filter, 50)            #提取边缘，后面数字是阈值
threshold_two(img04, 50)                              #转换二值图，后面数字也是阈值
clean_along_points(img04, 10)
show(img04,"IMG04",1)


img_part = []
for key in p_dirt.keys():                             #将图像分割成只有一个餐盘的部分，装入列表中
    img_part.append(get_points_img(img04, p_dirt[key]))

lll = 1
for part in img_part:
    part_cp = np.copy(part)
    show(part_cp, "part"+str(lll))
    lll +=1
    lines = Hough_line(part_cp, 50, 6)
    rect_lines = mf.check_rectangle(lines)     #备份矩形直线集
    if rect_lines:
        for rl in rect_lines:                     #画矩形
            for pl in rl:
                for l in pl:
                    delete_linepoints(part_cp, l, 255, 3)
        show(part_cp,"part_line"+str(lll),1)
    else:
        oval1 = ran_hough(part,5)
        oval1.print_fomula()
        img5 = make_image(oval1.points_on_oval())   #根据拟合出的椭圆画图
        show(img5, "endness"+str(lll))


# rgb_turn_hsi_img(img1)
# img_h,img_s,img_i = channal_div(img1)
# show(img1,"img2",0)
# show(img_h,"imgh",0)
# show(img_s,"imgs",0)
# show(img_i,"imgi",0)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img02)
cv2.waitKey(0)