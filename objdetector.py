#coding:utf-8

import numpy as np
import cv2


class objDetector:
    def __init__(self,image):
        self.image=image

    def detect(self,image=None):
        if image==None:
            image=self.image
        # cv2.imshow("image",image)
        # cv2.waitKey(1)
        cv2.imwrite("process\image.jpg",image)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray",gray)
        # cv2.waitKey(1)
        cv2.imwrite("process\gray.jpg", gray)

        gray = cv2.blur(gray, (5, 5))
        # cv2.imshow("gray-blur", gray)
        # cv2.waitKey(1)
        cv2.imwrite("process\grayblur.jpg", gray)

        # step2:用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域
        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

        # cv2.imshow("gradientX", cv2.convertScaleAbs(gradX))
        # cv2.waitKey(1)
        cv2.imwrite("process\gradX.jpg", cv2.convertScaleAbs(gradX))

        # cv2.imshow("gradientY", cv2.convertScaleAbs(gradY))
        # cv2.waitKey(1)
        cv2.imwrite("process\gradY.jpg", cv2.convertScaleAbs(gradY))

        # subtract the y-gradient from the x-gradient
        gradient = cv2.add(cv2.convertScaleAbs(gradX), cv2.convertScaleAbs(gradY))

        gradient = cv2.convertScaleAbs(gradient)
        # cv2.imshow("gradientXY", gradient)
        # cv2.waitKey(1)
        cv2.imwrite("process\gradientXY.jpg", gradient)

        # step3：去除图像上的噪声。首先使用低通滤泼器平滑图像（9 x 9内核）,这将有助于平滑图像中的高频噪声。
        # 低通滤波器的目标是降低图像的变化率。如将每个像素替换为该像素周围像素的均值。
        # 这样就可以平滑并替代那些强度变化明显的区域。
        # blur and threshold the image
        blurred = cv2.blur(gradient, (5, 5))
        # cv2.imshow("blurred模糊", blurred)
        # cv2.waitKey(1)
        cv2.imwrite("process\blurred.jpg", blurred)
        # 对模糊图像二值化。梯度图像中不大于90的任何像素都设置为0（黑色） 否则，像素设置为255（白色）
        (_, thresh) = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)  ####物体是浅色
        # (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)####物体是深色
        # cv2.imshow("thresh阈值", thresh)
        # cv2.waitKey(1)
        cv2.imwrite("process\thresh.jpg", thresh)

        # step4:在上图中我们看到蜜蜂身体区域有很多黑色的空余，我们要用白色填充这些空余，
        # 使得后面的程序更容易识别昆虫区域，这需要做一些形态学方面的操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))  # 定义模板
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)  # 闭运算,去除凹角
        # cv2.imshow("close闭运算", closed)
        # cv2.waitKey(1)
        cv2.imwrite("process\close.jpg", closed)

        # step5:从上图我们发现图像上还有一些小的白色斑点，这会干扰之后的昆虫轮廓的检测，要把它们去掉。
        # 分别执行4次形态学腐蚀与膨胀。
        # perform a series of erosions and dilations
        closed = cv2.erode(closed, None, iterations=1)  # 腐蚀
        # cv2.imshow("erode腐蚀1", closed)
        # cv2.waitKey(1)
        cv2.imwrite("process\erode1.jpg", closed)
        closed = cv2.erode(closed, None, iterations=1)  # 腐蚀
        # cv2.imshow("erode腐蚀2", closed)
        # cv2.waitKey(1)
        cv2.imwrite("process\erode2.jpg", closed)
        closed = cv2.erode(closed, None, iterations=1)  # 腐蚀
        # cv2.imshow("erode腐蚀3", closed)
        # cv2.waitKey(1)
        cv2.imwrite("process\erode3.jpg", closed)
        closed = cv2.erode(closed, None, iterations=1)  # 腐蚀
        # cv2.imshow("erode腐蚀4", closed)
        # cv2.waitKey(1)
        cv2.imwrite("process\erode4.jpg", closed)

        closed = cv2.dilate(closed, None, iterations=1)  # 膨胀
        # cv2.imshow("dilate膨胀1", closed)
        # cv2.waitKey(1)
        cv2.imwrite("process\dilate1.jpg", closed)
        closed = cv2.dilate(closed, None, iterations=1)  # 膨胀
        # cv2.imshow("dilate膨胀2", closed)
        # cv2.waitKey(1)
        cv2.imwrite("process\dilate2.jpg", closed)
        closed = cv2.dilate(closed, None, iterations=1)  # 膨胀
        # cv2.imshow("dilate膨胀3", closed)
        # cv2.waitKey(1)
        cv2.imwrite("process\dilate3.jpg", closed)
        closed = cv2.dilate(closed, None, iterations=1)  # 膨胀
        # cv2.imshow("dilate膨胀4", closed)
        # cv2.waitKey(1)
        cv2.imwrite("process\dilate4.jpg", closed)

        # step6：找出昆虫区域的轮廓。cv2.findContours()函数第一个参数是要检索的图片，必须是为二值图，即黑白的（不是灰度图），
        # 所以读取的图像要先转成灰度的，再转成二值图，我们在第三步用cv2.threshold()函数已经得到了二值图。
        # 第二个参数表示轮廓的检索模式，有四种：
        # cv2.RETR_EXTERNAL表示只检测外轮廓
        # cv2.RETR_LIST检测的轮廓不建立等级关系
        # cv2.RETR_CCOMP建立两个等级的轮廓，上面的一层为外边界，里面的一层为内孔的边界信息。如果内孔内还有一个连通物体，这个物体的边界也在顶层。
        # cv2.RETR_TREE建立一个等级树结构的轮廓。
        # 第三个参数为轮廓的近似方法
        # cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
        # cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
        # cv2.findContours()函数返回两个值，一个是轮廓本身，还有一个是每条轮廓对应的属性。
        # cv2.findContours()函数返回第一个值是list，list中每个元素都是图像中的一个轮廓，用numpy中的ndarray表示。
        # 每一个ndarray里保存的是轮廓上的各个点的坐标。我们把list排序，点最多的那个轮廓就是我们要找的昆虫的轮廓。
        # OpenCV中通过cv2.drawContours在图像上绘制轮廓。
        # 第一个参数是指明在哪幅图像上绘制轮廓
        # 第二个参数是轮廓本身，在Python中是一个list
        # 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓
        # 第四个参数是轮廓线条的颜色
        # 第五个参数是轮廓线条的粗细
        # cv2.minAreaRect()函数:
        # 主要求得包含点集最小面积的矩形，这个矩形是可以有偏转角度的，可以与图像的边界不平行。
        (_, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        # print c
        # compute the rotated bounding box of the largest contour
        rect = cv2.minAreaRect(c)
        # print "rect.type:" + str(type(rect))
        box = np.int0(cv2.boxPoints(rect))
        # print "box.type:" + str(type(box))

        # draw a bounding box arounded the detected barcode and display the image
        cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        # cv2.imshow("coutours原始轮廓", image)
        # cv2.waitKey(1)
        cv2.imwrite("process\contoursImage.jpg", image)

        # step7：裁剪。box里保存的是绿色矩形区域四个顶点的坐标。
        # 我将按下图红色矩形所示裁剪昆虫图像。找出四个顶点的x，y坐标的最大最小值。
        # 新图像的高=maxY-minY，宽=maxX-minX。
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        cropImg = image[y1:y1 + hight, x1:x1 + width]
        # cv2.imshow("cropImg", cropImg)
        # cv2.waitKey(1)
        cv2.imwrite("process\cropImg.jpg", cropImg)

        return x1,y1,width,hight