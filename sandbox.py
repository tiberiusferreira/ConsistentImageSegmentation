#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
from scipy import ndimage
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from robot_interaction_experiment.msg import Vision_Features
from robot_interaction_experiment.msg import Detected_Object
from robot_interaction_experiment.msg import Detected_Objects_List
import scipy
from skimage.feature import hog
from skimage import exposure
import cv2