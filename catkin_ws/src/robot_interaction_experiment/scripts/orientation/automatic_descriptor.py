#!/usr/bin/env python
import threading
from cv_bridge import CvBridge, CvBridgeError
from robot_interaction_experiment.msg import Audition_Features
import cv2
import string
import rospy
from robot_interaction_experiment.msg import Detected_Objects_List
import numpy as np
from collections import Counter
import time
import os
import pickle


def automatic_descriptor(label, detected_object):
    clr_hist = detected_object.features.colors_histogram
    clr_index = clr_hist.index(max(clr_hist))
    color = ''
    label = str(label)
    if 0 <= clr_index <= 12:
        color = 'yellow'
    elif 12 < clr_index <= 30:
        color = 'green'
    elif 30 < clr_index <= 49:
        color = 'cyan'
    elif 50 < clr_index <= 65:
        color = 'blue'
    elif 66 < clr_index <= 74:
        color = 'purple'
    else:
        color = 'red'
    color = str(color)
    label_n_clr = str(color) + ' ' + str(label)
    sentence_structs = list()
    sentence_structs.append('This is a ' + label_n_clr)
    sentence_structs.append('That is a ' + label_n_clr)
    sentence_structs.append('Here we have a ' + label_n_clr)
    sentence_structs.append('I can see a ' + label_n_clr)
    sentence_structs.append('In front of me there is ' + label_n_clr)
    sentence_structs.append('Here there is a ' + label + ' and its color is ' + color)
    sentence_structs.append('I see a ' + label + ' and its color is ' + color)
    filename = 'obj_description.pickle'
    if not os.path.isfile('obj_description.pickle'):
        with open(filename, 'w') as f:
            pickle.dump(sentence_structs, f)




