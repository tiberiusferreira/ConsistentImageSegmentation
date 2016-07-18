#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy
import cv2
import pylab as Plot
from sklearn.decomposition import PCA
from copy import copy
import time


from numpy import mean, cov, cumsum, dot, linalg, size, flipud

def princomp(A,numpc=0):
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
    [latent,coeff] = linalg.eig(cov(M))
    p = size(coeff,axis=1)
    idx = np.argsort(latent) # sorting the eigenvalues
    idx = idx[::-1]       # in ascending order
    # sorting eigenvectors according to the sorted eigenvalues
    coeff = coeff[:,idx]
    latent = latent[idx] # sorting eigenvalues
    if numpc < p and numpc >= 0:
        coeff = coeff[:,range(numpc)] # cutting some PCs if needed
    score = dot(coeff.T,M) # projection of the data in the new space
    return coeff,score,latent


def apply_pca(img):
    sklearn = 1
    opencv = 0
    data = list()
    axis = 2
    row, col = np.shape(img)
    offset = row / 2
    for x in range(row):
        for y in range(col):
            if img[x][y] > 0:
                data.append([x-(row/2), y-(col/2)])
    print (np.shape(data))
    if sklearn == 1:
        pca = PCA(n_components=2)
        pca.fit_transform(data)
        pca_comp = pca.components_ * 60
        print (pca_comp)
        print (pca.explained_variance_ratio_)
        print (pca)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.line(img, (int(pca_comp[1][0]+offset), int(pca_comp[1][1]+offset)),
                 (int(pca_comp[0][0]+offset), int(pca_comp[0][1]+offset)), (255, 0, 0), 3)
        # cv2.line(img, (offset, offset),
        #          (int(pca_comp[1][0]+offset), int(pca_comp[1][1]+offset)), (0, 255, 0), 5)
    # print (np.array(data).flatten())
    if opencv == 1:
        if axis == 2:
            pca_comp = princomp(np.array(data), 2)[0] * 60
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.line(img, (offset, offset),
                     (int(pca_comp[0][0]+offset), int(pca_comp[0][1]+offset)), (255, 0, 0), 5)
            cv2.line(img, (offset, offset),
                     (int(pca_comp[1][0]+offset), int(pca_comp[1][1]+offset)), (0, 255, 0), 5)
        if axis == 1:
            pca_comp = princomp(np.array(data), 1)[0] * 60
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.line(img, (offset, offset),
                     (int(pca_comp[0]+offset), int(pca_comp[1])+offset), (255, 0, 0), 5)
            # cv2.line(img, (offset, offset),
            #          (int(pca_comp[1][0]), int(pca_comp[1][1])), (0, 255, 0), 5)

    # cv2.line(img, (1, 1), (100, 10), (255, 0, 0), 5)
    for index, datas in enumerate(data):
        if index % 20 == 0:
            Plot.scatter(datas[0], datas[1])
    Plot.plot(np.linspace(offset, pca_comp[0][0], 5), np.linspace(offset, pca_comp[0][1], 5))
    # Plot.plot(np.linspace(offset, pca_comp[1][0], 5), np.linspace(offset, pca_comp[1][1], 5))
    Plot.show()
    cv2.imshow('RGB_img', img)
    cv2.waitKey(1)


if __name__ == '__main__':
    for imgs in ('bird_rot0.png', 'bird_rot1.png'):
        img = cv2.imread(imgs)
        w, l, d = np.shape(img)
        img = img[13:w - 5, 13:l - 8]
        img = cv2.resize(img, (400, 400))
        Sobelx = cv2.Sobel(img, -1, 1, 0, ksize=1)
        Sobely = cv2.Sobel(img, -1, 0, 1, ksize=1)
        # cv2.imshow('Sobelx', Sobelx)
        # cv2.imshow('Sobely', Sobelx)
        # print (Sobelx[100][100])
        Sobel = (Sobelx + Sobely)/2
        # Sobel = Sobelx**2 + Sobely**2
        # Sobel = np.sqrt(Sobel)
        # cv2.imshow('Sobel', Sobel)
        img = Sobel
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img_bgr8_clean_copy = cv2.GaussianBlur(img_bgr8_clean_copy, (5, 5), 0)
        ret3, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('After thresh', img)
        apply_pca(img)
        # cv2.waitKey(0)
