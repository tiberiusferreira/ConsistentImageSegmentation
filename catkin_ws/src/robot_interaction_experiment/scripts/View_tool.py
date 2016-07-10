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

img_shape_size = 30
max_histogram_size = 50
n_colors = 80 #40
font_size = 20
dist = 100



def NMF_dictionary_viewer(words_dictionary, NMF_dictionary):
    
    n_words = len(words_dictionary)
    # half_n_words = int(math.ceil(n_words/2.0))
    if issparse(NMF_dictionary):
        NMF_dictionary = NMF_dictionary.toarray()
    
    
    K = len(NMF_dictionary)
    img = np.zeros((10+img_shape_size+10+max_histogram_size+10+font_size*n_words+10, 10+(dist+10)*K+10, 3), dtype=np.uint8)
#     print "View NMF_dictionary.shape", NMF_dictionary.shape


    for Ki, NMF_histogram in enumerate(NMF_dictionary):
        shape_histogram, colors_histogram, words_histogram = NMF_histogram
        
        shape_data = np.array(shape_histogram)
        shape_data = shape_data * 255 / np.max(shape_data)
        shape_data = shape_data.astype(np.uint8)
        shape_data = cv2.cvtColor(shape_data, cv2.COLOR_GRAY2RGB)
#         print len(shape_data), img_shape_size
        img[10:10+img_shape_size, 10+Ki*(dist+10):10+Ki*(dist+10)+img_shape_size] \
                        = np.reshape(shape_data, (img_shape_size, img_shape_size, 3))
        
        for i_color in range(n_colors):
            color = float(i_color) / n_colors
            r, g, b = colorsys.hsv_to_rgb(color, 1.0, 1.0)
            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)
            cv2.line(img, (10+i_color+Ki*(dist+10), \
                           10+img_shape_size+10+max_histogram_size), \
                           (10+i_color+Ki*(dist+10), \
                10+img_shape_size+10+max_histogram_size-int(colors_histogram[i_color] * max_histogram_size)), \
                     (b, g, r), 1)
        
#         print "View words_histogram", words_histogram
        for i, val in enumerate(words_histogram):
            word = words_dictionary[i]
            lum = min(int(val * 255), 255)  #  change the highlight of words
            cv2.putText(img, word, (10+Ki*(dist+10), \
                                    10+img_shape_size+10+max_histogram_size+10+font_size+i*font_size), \
                        cv2.FONT_HERSHEY_PLAIN, font_size/20.0, (lum, lum, lum))
            # 10+150*(i/half_n_words)
        
        
        cv2.putText(img, str(Ki), (10+60+Ki*(dist+10), 10+font_size), \
                    cv2.FONT_HERSHEY_PLAIN, font_size/20.0, (255, 255, 255))
    
    return img
