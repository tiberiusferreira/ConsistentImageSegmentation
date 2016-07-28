#!/usr/bin/env python
import threading
from cv_bridge import CvBridge, CvBridgeError
from robot_interaction_experiment.msg import Audition_Features
import cv2
from preprocessRecordings import preprocessfile
import rospy
from View_tool import NMF_dictionary_viewer
import numpy as np
import sys
import joblib

def create_hist_matrix(file):
    lines_since_obj = -1000000
    global matrice_hist
    matrice_hist = list()
    concat = list()
    global dict
    dict= list()
    got_dic = 0
    for line_num, line in enumerate(file):
        if lines_since_obj == 1 and got_dic == 0:
            for string in line.strip().split(', '):
                dict.append(string)
                got_dic = 1
        if lines_since_obj == 4 or lines_since_obj == 5 or lines_since_obj == 6:  # Respectively HoG, Color and Words
            hog_cont = 0
            for string in line.strip().split(', '):
                string = float(string)
                concat.append(string)
                hog_cont += 1
            if lines_since_obj == 4:
                while hog_cont < 900:
                    concat.append(0)
                    hog_cont += 1
        lines_since_obj += 1
        if '#Object' in line:
            if concat:
                matrice_hist.append(concat)
            concat = []
            lines_since_obj = 1
    matrice_hist.append(concat)





if __name__ == '__main__':
    if len(sys.argv) > 2:
        print 'Pass only the (path) to the recordings file.\n'
        exit(1)
    if len(sys.argv) < 2:
        print 'I need the path where ExperimentDataLog.txt and HoG_imgs.pickle are.'
        exit(1)
    print (sys.argv[1])
    try:
        f = open(sys.argv[1] + '/ExperimentDataLog.txt', 'r', 0)
    except IOError:
        print ("Could not open file " + str(sys.argv[1] + "!"))
        exit(1)
    create_hist_matrix(f)
    global matrice_hist
    matrice_hist = np.array(matrice_hist)
    n_hog = 900
    n_color = 80

    ####  to record the img before each training  ########
    global dict

    words_dictionary_visual = dict
    Xres = matrice_hist
    # print matrice_hist
    hog_n_color = joblib.load(sys.argv[1] + '/hog_n_color.pickle')

    NMF_dictionary_visual = np.array([np.array([Xres[0, n_hog:n_color + n_hog],
                                                Xres[0, n_color + n_hog:]])])
    for i in range(1, len(Xres)):
        values = np.array([np.array([Xres[i, n_hog:n_color + n_hog],
                                     Xres[i, n_color + n_hog:]])])
        NMF_dictionary_visual = np.vstack([NMF_dictionary_visual, values])

    img_iteration = NMF_dictionary_viewer(words_dictionary_visual, hog_n_color, NMF_dictionary_visual)
    cv2.imshow('Img', img_iteration)
    cv2.waitKey(0)
    cv2.imwrite(sys.argv[1] + 'Exp_data_visualization.png', img_iteration)

