#coding:utf-8
import numpy as np
import cv2

class colorDetector:
    def __init__(self,image,lower=None,upper=None):
        # 设定红色阈值，HSV空间
        if lower==None:
            self.lower=np.array([170, 100, 100])
        if upper==None:
            self.upper=np.array([179, 255, 255])
        self.image=image
    def detect(self):
        frame=self.image
        #转到HSV空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #根据阈值构建掩膜
        mask = cv2.inRange(hsv, self.lower, self.upper)
        #腐蚀操作
        mask = cv2.erode(mask, None, iterations=2)
        #膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
        mask = cv2.dilate(mask, None, iterations=2)
        #cv2.imshow("dilate",mask)
        #cv2.waitKey(0)
        #轮廓检测
        (_,cnts,_)= cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #初始化瓶盖圆形轮廓质心
        center = None
        #如果存在轮廓
        if len(cnts) > 0:
            #找到面积最大的轮廓
            c = max(cnts, key = cv2.contourArea)
            #确定面积最大的轮廓的外接圆
            rect= cv2.minAreaRect(c)
            box=cv2.boxPoints(rect)
            box=np.int0(box)
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)
            hight = y2 - y1
            width = x2 - x1
            ((x, y), radius) =cv2.minEnclosingCircle(c)

            #计算轮廓的矩
            M = cv2.moments(c)
            # print (x,y),radius
            #计算质心
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            p1=(center[0]-int(radius),center[1]-int(radius))
            p2=(center[0]+int(radius),center[1]+int(radius))

            return x1,y1,width,hight

