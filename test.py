#coding:utf-8
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
def show(img, code=cv2.COLOR_BGR2RGB):
    cv_rgb = cv2.cvtColor(img, code)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(cv_rgb)
    fig.show()

def cv_distance(P, Q):
    return int(math.sqrt(pow((P[0] - Q[0]), 2) + pow((P[1] - Q[1]),2)))

def createLineIterator(P1, P2, img):
   imageH = img.shape[0]
   imageW = img.shape[1]
   P1X = P1[0]
   P1Y = P1[1]
   P2X = P2[0]
   P2Y = P2[1]

   #difference and absolute difference between points
   #used to calculate slope and relative location between points
   dX = P2X - P1X
   dY = P2Y - P1Y
   dXa = np.abs(dX)
   dYa = np.abs(dY)

   #predefine numpy array for output based on distance between points
   itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
   itbuffer.fill(np.nan)

   #Obtain coordinates along the line using a form of Bresenham's algorithm
   negY = P1Y > P2Y
   negX = P1X > P2X
   if P1X == P2X: #vertical line segment
       itbuffer[:,0] = P1X
       if negY:
           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
       else:
           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
   elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
       else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
   else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX.astype(np.float32)/dY.astype(np.float32)
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
       else:
           slope = dY.astype(np.float32)/dX.astype(np.float32)
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

   #Remove points outside of image
   colX = itbuffer[:,0]
   colY = itbuffer[:,1]
   itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

   #Get intensities from img ndarray
   itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

   return itbuffer


def check(a, b):
    # 存储 ab 数组里最短的两点的组合
    s1_ab = ()
    s2_ab = ()
    # 存储 ab 数组里最短的两点的距离，用于比较
    s1 = np.iinfo('i').max
    s2 = s1
    for ai in a:
        for bi in b:
            d = cv_distance(ai, bi)
            if d < s2:
                if d < s1:
                    s1_ab, s2_ab = (ai, bi), s1_ab
                    s1, s2 = d, s1
                else:
                    s2_ab = (ai, bi)
                    s2 = d
    (a1, b1) = s1_ab
    (a2, b2) = s2_ab
    # 将最短的两个线画出来
    # cv2.line(draw_img, a1, b1, (0,0,255), 3)
    # cv2.line(draw_img, a2, b2, (0,0,255), 3)
    return s1_ab,s2_ab


def mycheck(a, b,img_gray,contours,ix,jx):

    contour_all = []
    for point in b:
        contour_all.append(point)
    for point in a:
        contour_all.append(point)
    contour_all = np.array(contour_all)
    rect = cv2.minAreaRect(contour_all)
    box = cv2.boxPoints(rect)
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = int(min(Xs))
    x2 = int(max(Xs))
    y1 = int(min(Ys))
    y2 = int(max(Ys))
    hight = y2 - y1
    width = x2 - x1
    cropImg = img_gray[y1:y1 + hight, x1:x1 + width]
    # cv2.imshow("region",cropImg)
    # cv2.waitKey(0)
    avg=np.average(cropImg)
    th, bi_img = cv2.threshold(img_gray, avg, 255, cv2.THRESH_BINARY)
    # cv2.imshow("bbbb",bi_img)
    # cv2.waitKey(0)
    # 存储 ab 数组里最短的两点的组合
    s1_ab = ()
    s2_ab = ()
    # 存储 ab 数组里最短的两点的距离，用于比较
    s1 = np.iinfo('i').max
    s2 = s1
    for ai in a:
        for bi in b:
            d = cv_distance(ai, bi)
            if d < s2:
                if d < s1:
                    s1_ab, s2_ab = (ai, bi), s1_ab
                    s1, s2 = d, s1
                else:
                    s2_ab = (ai, bi)
                    s2 = d
    (a1, b1) = s1_ab
    (a2, b2) = s2_ab
    a1 = (a1[0] + (a2[0]-a1[0])*1/14, a1[1] + (a2[1]-a1[1])*1/14)
    b1 = (b1[0] + (b2[0]-b1[0])*1/14, b1[1] + (b2[1]-b1[1])*1/14)
    a2 = (a2[0] + (a1[0]-a2[0])*1/14, a2[1] + (a1[1]-a2[1])*1/14)
    b2 = (b2[0] + (b1[0]-b2[0])*1/14, b2[1] + (b1[1]-b2[1])*1/14)
    lineiter1=createLineIterator(a1,b1,bi_img)
    bivalue1= lineiter1[:,2]
    try:
        flag1=isTimingPattern(bivalue1)
    except BaseException:
        img = img_gray.copy()
        cv2.line(img, a2, b2, 0, 2)
        cv2.imshow("line1", img)
        img_dc =img_gray.copy()
        cv2.drawContours(img_dc, contours, ix, 255, 3)
        cv2.drawContours(img_dc, contours, jx, 255, 3)
        cv2.imshow("box"+str(i),img_dc)
        cv2.waitKey(0)
    # flag1= isTimingPattern(bivalue1)
    # if flag1:
    #     img = img_gray.copy()
    #     cv2.line(img, a1, b1, 0, 2)
    #     cv2.imshow("line1", img)
    #     cv2.waitKey(0)
    lineiter2 = createLineIterator(a2, b2, bi_img)
    bivalue2 = lineiter2[:, 2]
    try:
        flag2=isTimingPattern(bivalue2)
    except BaseException:
        pass
        # img = img_gray.copy()
        # cv2.line(img, a2, b2, 0, 2)
        # cv2.imshow("line2", img)
        # img_dc = img_gray.copy()
        # cv2.drawContours(img_dc, contours, ix, 0, 3)
        # cv2.drawContours(img_dc, contours, jx, 0, 3)
        # cv2.imshow("box" + str(i), img_dc)
        # cv2.waitKey(0)
    # if flag2:
    #     img = img_gray.copy()
    #     cv2.line(img, a2, b2, 0, 2)
    #     cv2.imshow("line2", img)
    #     cv2.waitKey(0)
    return flag1 or flag2


def isTimingPattern(line):
    # 除去开头结尾的白色像素点
    # try:
    #     while line[0] != 0:
    #         line = line[1:]
    #     while line[-1] != 0:
    #         line = line[:-1]
    # except BaseException:
    #     return  False
    # 计数连续的黑白像素点
    c = []
    count = 1
    l = line[0]
    for p in line[1:]:
        if p == l:
            count = count + 1
        else:
            c.append(count)
            count = 1
        l = p
    c.append(count)
    # 如果黑白间隔太少，直接排除
    if len(c) < 5:
        return False
    # 计算方差，根据离散程度判断是否是 Timing Pattern
    avg=sum(c)/len(c)
    # print np.var(c)
    return np.std(c) < avg*0.8

def imgresize(img,radio):
    img=cv2.resize(img,(int(img.shape[1]/radio),int(img.shape[0]/radio)))
    return img

