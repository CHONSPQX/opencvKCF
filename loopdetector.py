# coding:utf-8
import math
import cv2
from matplotlib import pyplot as plt
from test import *
import numpy as np



def lodetector(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 100, 200)
    _, bi_img = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    img_fc, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy[0]
    found = set()
    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            k = hierarchy[k][2]
            c = c + 1
        if c >= 2:
            found.add((i,c))
    try:
        found_max=max(found,key=lambda i:i[1])[0]
        print found
        rect = cv2.minAreaRect(contours[found_max])
        box = cv2.boxPoints(rect)
        box = np.array(box)
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        x2 = max(Xs)
        y1 = min(Ys)
        y2 = max(Ys)
        hight = y2 - y1
        width = x2 - x1
        print x1,y1,width,hight
    except BaseException:
        x1=y1=width=hight=1
    return x1, y1, width, hight



def all(img_name, resize_flag=False):
    print img_name
    img = cv2.imread(img_name)
    # print img.shape
    if resize_flag:
        img = imgresize(img,3)
    # print img.shape
    # cv2.imshow("img",img)
    # cv2.waitKey(0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("img_gray",img_gray)
    # cv2.waitKey(0)
    img_gb = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # cv2.imshow("img_gb",img_gb)
    # cv2.waitKey(0)
    edges = cv2.Canny(img_gray, 100, 200)
    # cv2.imshow("edges",edges)
    # cv2.waitKey(0)
    _, bi_img = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)
    # print bi_img
    # cv2.imshow("bi_img",bi_img)

    img_fc, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print
    hierarchy = hierarchy[0]
    # print len(contours)
    found = set()
    layer = []
    for i in range(len(contours)):
        k = i
        c = 0
        while hierarchy[k][2] != -1:
            # print hierarchy[k]
            # print hierarchy[k][2]
            # print hierarchy[k][0]
            k = hierarchy[k][2]
            c = c + 1
        layer.append((i, c))
        if c >= 4:
            found.add(i)

    ffound = set()

    for i in found:
        for j in found:
            if i == j:
                continue
            if (hierarchy[j][2] == i):
                ffound.add(i)

    found = found - ffound

    found = list(found)

    layer = np.array(layer)

    # for i in found:
    #     img_dc = img.copy()
    #     cv2.drawContours(img_dc, contours, i, (0, 255, 0), 3)
    #     cv2.imshow("img"+str(i),img_dc)
    #     cv2.waitKey(0)

    draw_img = img.copy()

    for i in found:
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(draw_img, [box], 0, (0, 0, 255), 2)

    # cv2.imshow("draw_img",draw_img)
    # cv2.waitKey(0)

    boxes = []
    for i in found:
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box = map(tuple, box)
        # print box
        boxes.append(box)

    print "boxes.num: " + str(len(boxes))

    valid = set()

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if mycheck(boxes[i], boxes[j], img_gray, contours, i, j):
                valid.add(i)
                valid.add(j)

    # cv2.imshow("line",draw_img)
    # cv2.waitKey(0)
    # print bi_img


    print "valid: ", valid

    contour_all = []
    while len(valid) > 0:
        c = valid.pop()
        # print c
        for point in boxes[c]:
            contour_all.append(point)
    contour_all = np.array(contour_all)
    # print contour_all.shape

    rect = cv2.minAreaRect(contour_all)
    box = cv2.boxPoints(rect)
    box = np.array(box)
    draw_img = img.copy()
    # cv2.polylines(draw_img, np.int32([box]), True, (0, 0, 255), 3)
    # cv2.imshow("outcome", draw_img)
    # cv2.waitKey(0)


# all('1.jpg')
# all('2.jpg', True)
# all('3.jpg')
# all('6.jpg',1)
# all('5.jpg',True)
# cv2.waitKey(0)
