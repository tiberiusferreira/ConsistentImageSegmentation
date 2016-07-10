#!/usr/bin/env python
import threading
from cv_bridge import CvBridge, CvBridgeError
from robot_interaction_experiment.msg import Audition_Features
import cv2
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import neighbors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import rospy
import pylab as pl
from robot_interaction_experiment.msg import Detected_Objects_List

'''Creates a HoG Confusion Matrix from the /detected_objects_list topic. The user gives the class of the object to be
captured and then the program captures it's info from the topic'''

hog_list = list()
hog_ori_list = list()


def callback_rgb(data):
    det_objects = data.detected_objects_list
    global num_obj
    global last_img
    global last_hog

    num_obj = len(det_objects)
    if num_obj > 0:
        last_hog = det_objects[0].features.hog_histogram
        try:
            last_img = CvBridge().imgmsg_to_cv2(det_objects[0].image)
        except CvBridgeError, e:
            print e
            return
        cv2.imshow('a', last_img)
        cv2.waitKey(1)



if __name__ == '__main__':
    rospy.init_node('HoGConfMatrixCapturer', anonymous=True)
    object_sub = rospy.Subscriber("detected_objects_list", Detected_Objects_List, callback_rgb)
    global num_obj
    labels = []
    num_obj = 0
    global last_hog
    clf = svm.SVC()
    colors = []

    while 1:
        try:
            mode = str(raw_input('Input:'))
        except KeyboardInterrupt:
            print "Cancelled"
        if mode == 'q':
            exit(0)
        if mode == 'train':
            print 'Ok training!\n'
            print len(hog_ori_list)
            clf.fit(hog_ori_list, labels)
            trained = 1
            print 'New batch ready!'
            continue
        if num_obj == 0:
            print "No objects detected."
            continue
        if num_obj > 1:
            print "Too many detected objects"
            continue
        if mode == 'g':
            print clf.predict([last_hog])
            continue
        if mode == '-1':
            cov = np.cov(hog_list)
            print cov
            print "Max = " + str(np.max(cov))
            print "Min = " + str(np.min(cov))
            print labels

            data = np.flipud(cov)  # 25x25 matrix of values
            pl.pcolor(data)
            pl.colorbar()
            pl.show()

            continue
        if labels:
            if mode not in labels:
                colors.append(np.random.rand())
        labels.append(mode)

        hog_list.append(last_hog)
        hog_ori_list.append(last_hog)
