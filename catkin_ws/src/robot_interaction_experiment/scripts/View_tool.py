#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import colorsys
import math
import sys
import os
import scipy
from scipy import io
from scipy.sparse import issparse

img_size = 80
max_histogram_size = 50
n_colors = 80
font_size = 20
horz_dist = 100


def NMF_dictionary_viewer(words_dictionary, hog_n_color, NMF_dictionary):
    
    n_words = len(words_dictionary)
    if issparse(NMF_dictionary):
        NMF_dictionary = NMF_dictionary.toarray()
    
    K = len(NMF_dictionary)
    img = np.zeros((80 + img_size + max_histogram_size  + font_size * n_words, 10 + (horz_dist + 10) * K + 10, 3), dtype=np.uint8)

    for Ki, NMF_histogram in enumerate(NMF_dictionary):
        colors_histogram, words_histogram = NMF_histogram
        hog_img, color_img = hog_n_color[Ki]
        hog_img = cv2.cvtColor(hog_img, cv2.COLOR_GRAY2BGR)
        hog_img = cv2.resize(hog_img, (img_size, img_size))
        img[10:10 + img_size, 10 + Ki * (horz_dist + 10):10 + Ki * (horz_dist + 10) + img_size] = np.reshape(hog_img, (img_size, img_size, 3))

        color_img = cv2.resize(color_img, (img_size, img_size))
        img[20 + img_size:20 + 2 * img_size, 10 + Ki * (horz_dist + 10):10 + Ki * (horz_dist + 10) + img_size] = np.reshape(color_img, (img_size, img_size, 3))

        
        for i_color in range(n_colors): # n_colors = 80
            color = float(i_color) / n_colors
            r, g, b = colorsys.hsv_to_rgb(color, 1.0, 1.0)
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)
            cv2.line(img, (10 + i_color + Ki * (horz_dist + 10), \
                           15 + img_size + 50 + max_histogram_size), \
                     (10 + i_color + Ki * (horz_dist + 10), \
                      15 + img_size + 50 + max_histogram_size - int(colors_histogram[i_color] * max_histogram_size)), \
                     (b, g, r), 1)
        
#         print "View words_histogram", words_histogram
        for i, val in enumerate(words_histogram):
            word = words_dictionary[i]
            lum = min(int(val * 255), 255)  #  change the highlight of words
            cv2.putText(img, word, (10 + Ki * (horz_dist + 10), \
                                    10 + img_size + 40 + max_histogram_size + 10 + font_size + i * font_size), \
                        cv2.FONT_HERSHEY_PLAIN, font_size / 20.0, (lum, lum, lum))
            # 10+150*(i/half_n_words)
        
        
        cv2.putText(img, str(Ki), (10 + img_size + Ki * (horz_dist + 10), 10 + font_size), \
                    cv2.FONT_HERSHEY_PLAIN, font_size / 20.0, (255, 255, 255))

    
    return img
