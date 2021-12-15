import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import cv2 as cv


def draw_label(path):
    tree = ET.parse(path)
    img_out = np.zeros(shape=(1024, 1280))
    img_list_x = []
    img_list_y = []
    for elem in tree.iterfind('object'):
        mylist_x = []
        mylist_y = []
        # print(elem.tag, elem.attrib)
        for elem_1 in elem.iterfind('polygon/pt'):
            object_x = elem_1.find("x").text
            object_y = elem_1.find("y").text
            x = int(object_x)
            y = 1024 - int(object_y)
            if x < 0:
                x = 0
            if x > 1279:
                x = 1279
            if y < 0:
                y = 0
            if y > 1023:
                y = 1023
            mylist_x.append(x)
            mylist_y.append(y)
            img_list_x.append(x)
            img_list_y.append(y)
            img_out.itemset((y, x), 255)
        mylist = list(zip(mylist_x, mylist_y))
        pts = np.array(mylist, np.int32)
        cv.polylines(img_out, [pts], True, (255, 255, 255), 2)  # 画线
        cv.fillPoly(img_out, [pts], (255, 255, 255))  # 填充内部
    Alllist = list(zip(img_list_x, img_list_y))  # 统计标注点
    # cv.imwrite('./picture/label.png', img_out)
    return img_out


def getlabel(path):
    img1 = draw_label(path)
    list_out = np.zeros(shape=(1024, 1280))

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if img1[i, j] == 255:
                list_out[i, j] = 1
    return list_out

