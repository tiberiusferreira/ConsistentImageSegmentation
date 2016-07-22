#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import pylab as Plot
from sklearn.decomposition import PCA
from orientation import pca, features_based
import os
if __name__ == '__main__':
    # PREFIX = '../rapport_fig/'
    PREFIX = '../TST_RES/'
    TST_PATH = 'TST_IMGS/'
    for imgs in os.listdir(TST_PATH):
        if not imgs.endswith(".png") or not imgs.startswith("cake"):
            continue
        print ('Doing: ' + str(imgs))
    # for imgs in ('bird_rot0.png', 'bird_rot1.png', 'bird_rot2.png', 'bird_rot3.png', 'fish_rot0.png', 'fish_rot1.png', 'fish_rot2.png', 'fish_rot3.png', 'clipincl1.png', 'clipincl2.png', 'clip2.png', 'clip2inv.png', 'cake1.png', 'cake2.png','cake3.png', 'frog_rot0.png', 'frog_rot1.png', 'frog_rot2.png', 'frog_rot3.png'):
        img = cv2.imread(TST_PATH + imgs)
        w, l, d = np.shape(img)
        # img = img[15:w - 7, 15:l - 10]
        img = cv2.resize(img, (400, 400))
        img_rotated = pca.apply_pca_rotation(img.copy())
        imgs_sift = features_based.sifts_up(img_rotated, 0)
        imgs_cnt = features_based.countourpnts_up(img_rotated)
        # cv2.imwrite('Original_' + str(imgs), pca.draw_axis(img, False))

        cv2.imwrite(PREFIX + 'Thresh_' + str(imgs), pca.threshold_img(img))
        cv2.imwrite(PREFIX + 'PCA_' + str(imgs), pca.draw_axis(img_rotated, False))
        # cv2.imshow('PCA_' + str(imgs), img_rotated)
        cv2.imwrite(PREFIX + 'Sift_' + str(imgs), imgs_sift)
        cv2.imwrite(PREFIX + 'Cnts_' + str(imgs), imgs_cnt)
        # cv2.waitKey(0)

        # cv2.imwrite('Rot ' + str(imgs), img_rotated)
        # cv2.imshow('Rot ' + str(imgs), img_rotated)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        # img_axis = pca.draw_axis(img.copy(), dgb=False)
        # pca.plot_img_data(pca.threshold_img(img))

        cv2.imwrite(PREFIX + 'Axis_' + str(imgs), pca.draw_axis(img, False))
        # cv2.imshow('Axis ' + str(imgs), img_axis)
        # cv2.waitKey(0)
    cv2.waitKey(0)

    #     print ('Point = ' + str(point))
    #     print ('Angle= ' + str(np.arctan2(point[0], point[1])))
    #
    #     print ('PCA components = ')
    #     print (pca_comp)
    #     print (pca.explained_variance_ratio_)
    #     print (pca)
    #     img_bgr8 = cv2.cvtColor(img_bgr8, cv2.COLOR_GRAY2BGR)
    #     # print (int(pca_comp[1][1]))
