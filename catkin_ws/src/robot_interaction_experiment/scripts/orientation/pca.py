#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pylab as Plot
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# receives a bgr8 thresholded image and returns the eigenvectors resulting of the pca
def apply_pca(img_bgr8):
    img_bgr8 = cv2.cvtColor(img_bgr8, cv2.COLOR_BGR2GRAY)
    row, col = np.shape(img_bgr8)
    x_data = list()
    y_data = list()
    for x in range(row):
        for y in range(col):
            if img_bgr8[x][y] > 0:
                x_data.append(x)
                y_data.append(y)
    y_data.reverse()
    array = np.c_[x_data, y_data]
    pca = PCA(n_components=2)
    pca.fit_transform(array)
    pca_score = pca.explained_variance_ratio_
    pca_comp = pca.components_ * 150
    # print (pca_score)
    pca_comp[0][0] = pca_comp[0][0] * pca_score[1]
    pca_comp[0][1] = pca_comp[0][1] * pca_score[1]
    pca_comp[1][0] = pca_comp[1][0] * pca_score[0]
    pca_comp[1][1] = pca_comp[1][1] * pca_score[0]
    return pca_comp


def plot_img_data(img_bgr8):
    img_bgr8 = cv2.cvtColor(img_bgr8, cv2.COLOR_BGR2GRAY)
    row, col = np.shape(img_bgr8)
    x_data = list()
    y_data = list()
    for x in range(row):
        for y in range(col):
            if img_bgr8[x][y] > 0:
                x_data.append(x)
                y_data.append(y)
    y_data.reverse()
    for index, data in enumerate(x_data):
        if index % 5 == 0:
            plt.scatter(y_data[index], data)
    plt.xlim(0, 400)
    plt.ylim(0, 400)
    plt.show()



def get_sobel(img_bgr8):
    # cv2.imshow('Got in Sobel CLR', img_bgr8)
    # cv2.waitKey(0)
    Sobel = 0
    for colors in range(3):
        one_channel = img_bgr8[:,:,colors]

        # cv2.imshow('Single Channel', one_channel)
        # cv2.waitKey(0)
        sobelx64f = cv2.Sobel(one_channel.copy(), cv2.CV_64F, 1, 0, ksize=1)
        abs_sobelx64f = np.absolute(sobelx64f)

        sobel_8ux = np.uint8(abs_sobelx64f)

        # sobel_8ux *= 255 / np.amax(sobel_8ux)


        sobely64f = cv2.Sobel(one_channel.copy(), cv2.CV_64F, 0, 1, ksize=1)
        abs_sobely64f = np.absolute(sobely64f)
        sobel_8uy = np.uint8(abs_sobely64f)
        # cv2.imshow('Sobel Y', sobel_8uy)
        # cv2.waitKey(0)

        # cv2.imshow('Sobel X', sobel_8ux)
        # cv2.waitKey(0)
        Sobel_ac = np.divide(sobel_8ux, 2) + np.divide(sobel_8uy, 2)
        Sobel_ac *= 255 / np.amax(Sobel_ac)
        # cv2.imshow('Sobel part', Sobel_ac)
        # cv2.waitKey(0)
        Sobel_ac = np.divide(Sobel_ac, 3)
        Sobel += Sobel_ac



    Sobel *= 255/np.amax(Sobel)
    # cv2.imshow('Got out Thre', Sobel)
    # cv2.waitKey(0)
    return Sobel

# gets img and returns the thresholded image after Sobel, the image is returned in bgr8 format
def threshold_img(img_bgr8):

    Sobel = get_sobel(img_bgr8)

    ret3, thresh = cv2.threshold(Sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    bgr8 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    return bgr8

# gets a matrix 2x2 representing 2 vectors and returns the point representing the biggest
def get_biggest_vector(vectors):
    if np.sqrt(vectors[0][0] ** 2 + vectors[0][1] ** 2) > np.sqrt(vectors[1][0] ** 2 + vectors[1][1] ** 2):
        point = (vectors[0][0], vectors[0][1])
    else:
        point = (vectors[1][0], vectors[1][1])
    return point


# get an bgr8 image and returns it rotated 90 degrees
def rotate_90(img_bgr):
    rows, cols, depth = img_bgr.shape
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    img_bgr = cv2.warpAffine(img_bgr, m, (cols, rows), borderMode=cv2.BORDER_WRAP)
    return img_bgr

# gets an bgr8 img and returns the image rotated so with the longest axis closer to the vertical
def apply_pca_rotation(img_bgr8):
    img_bgr8_thresh = threshold_img(img_bgr8)
    all_vectors = apply_pca(img_bgr8_thresh)
    point = get_biggest_vector(all_vectors)
    if abs(point[0]) > abs(point[1]):
        img_bgr8 = rotate_90(img_bgr8)
    return img_bgr8

# gets an bgr8 img and returns it with pca axis drawn
def draw_axis(img, dgb):
    row, col, d = np.shape(img)
    offsetx = row / 2
    offsety = row / 2
    thresholded_img = threshold_img(img)

    all_vectors = apply_pca(thresholded_img)
    img_line = img.copy()
    cv2.line(img_line, (offsetx, offsety),
             (int(all_vectors[1][0]) + offsetx, int(all_vectors[1][1]) + offsety), (255, 0, 0), 3)
    cv2.line(img_line, (offsetx, offsety),
             (int(all_vectors[0][0]) + offsetx, int(all_vectors[0][1]) + offsety), (255, 0, 0), 3)
    if dgb:
        cv2.imshow('Thresholded ' + str(all_vectors[1][0]), thresholded_img)
        cv2.waitKey(0)
        Sobel = get_sobel(img)

        cv2.imshow('Sobel + ' + str(all_vectors[1][0]), Sobel)
        cv2.waitKey(0)

    return img_line

