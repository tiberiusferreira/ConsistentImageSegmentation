#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy
import cv2
if __name__ == '__main__':
    img = cv2.imread('NoLightCut.png')
    # img = np.array(img, dtype=np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.Canny(img, 50, 200, 3)
    cv2.imshow('Img', img)
    cv2.waitKey(0)
    im, cnts, hir = cv2.findContours(img.copy(), 1, 2)
    # print (im)
    cntss = list()
    for cnt in cnts:
        if cv2.contourArea(cnt) > 10:
            cntss.append(cnt)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(img, cntss, -1, (255, 0, 0), 2)
    cv2.imshow('Img', img)
    cv2.waitKey(0)
    cv2.imwrite('notebook1CutSeg.png',img)
