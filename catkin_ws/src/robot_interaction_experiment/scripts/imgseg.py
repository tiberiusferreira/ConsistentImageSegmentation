#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import rospy
import numpy as np
from scipy import ndimage
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from robot_interaction_experiment.msg import Vision_Features
from robot_interaction_experiment.msg import Detected_Object
from robot_interaction_experiment.msg import Detected_Objects_List
from robot_interaction_experiment.msg import Audition_Features
from collections import Counter
import joblib
import os
import cv2
import pickle
import Queue
import time
import pylab as plot
import tsne
import copy
import math
import threading
from orientation import pca, machinelearning, min_area_triang, features_based
from skimage.feature import hog
from skimage import exposure
import colorsys
import matplotlib.pyplot as plt
import orientation.automatic_descriptor

####################
# Constants max HoG size = 900  #
####################

'''This is the main script used to segment the objects. It makes use of one of the scripts placed in the orientation folder
 to find the proper image orientation.
'''

MAIN_WINDOW_NAME = 'Segmentator'
RGB_WINDOW_NAME = 'RGB'
DEPTH_WINDOW_NAME = 'DEPTH'

MIN_AREA = 8500  # minimal area to consider a desirable object
MAX_AREA = 16000  # maximal area to consider a desirable object
MIN_CNT_LENGTH = 400  # minimal contour lenght to be consided a desirable object
MAX_CNT_LENGTH = 640  # maximal contour lenght to be consided a desirable object
N_COLORS = 80  # number of colors to consider when creating the image descriptor


####################
# Global variables #
####################
using_VGA = 0
got_speech = 0
depth_img_avg = np.zeros((100, 100))
img_bgr8_clean = np.zeros((100, 100))
show_color = False
show_depth = False
clr_timestamp = 0
depth_timestamp = 0
last_clr_timestamp = 0
last_depth_timestamp = 0
val_depth_capture = 0.03
rgb_lock = threading.Lock()
depth_lock = threading.Lock()
treatment_lock = threading.Lock()

# is called each time a new image is received by ROS
def callback_rgb(msg):
    rgb_lock.acquire()
    global clr_timestamp, img_bgr8_clean, using_VGA
    try:
        img = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print(e)
        rgb_lock.release()
        return
    height, width, depth = np.shape(img)
    if width == 1280 and height == 1024:
        using_VGA = 0
        img = img[32:992, 0:1280]  # crop the image because it does not have the same aspect ratio of the depth one
    elif not width == 640 or not height == 480:
        using_VGA = 1
        print('Neither SXGA (1280x1024) or VGA (640x480), resolution = ' + str(width) + 'x' + str(height) + '.')
        print('Resolution not supported.')
        exit(1)
    clr_timestamp = msg.header.stamp
    img_bgr8_clean = img
    rgb_lock.release()


# is called each time a new depth image is received by ROS
def callback_depth(msg):
    depth_lock.acquire()
    # treating the image containing the depth data
    global depth_img_avg, depth_timestamp, clr_timestamp, last_depth_imgs
    # getting the image
    if 'last_depth_imgs' not in globals():
        last_depth_imgs = list()
    try:
        img = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        print(e)
        return
    # removing eventual nan values
    cleanimage = clean_inf_n_nan(img, 255)
    if clr_timestamp == 0:
        depth_lock.release()
        return
    height, width, depth = np.shape(img_bgr8_clean)
    # getting color and depth to the same size
    cleanimage = cv2.resize(cleanimage, (width, height))
    # saving a history of depth images and taking the average of 4 of them as the final one
    # cant do better than that because the color and depth images are not well syncronized
    last_depth_imgs.append(copy.copy(cleanimage))
    if len(last_depth_imgs) > 4:
        last_depth_imgs = last_depth_imgs[1:]
    else:
        depth_lock.release()
        return
    depth_img_avg = np.zeros(np.shape(last_depth_imgs[0]), dtype=np.float32)
    for depth_img in last_depth_imgs:
        depth_img_avg += depth_img
    depth_img_avg /= 4
    depth_timestamp = msg.header.stamp
    depth_lock.release()


def show_clr_imgs(b):
    global show_color
    if b == 1:
        show_color = True
    else:
        show_color = False


def show_depth_imgs(b):
    global show_depth
    if b == 1:
        show_depth = True
    else:
        show_depth = False


def show_objs(b):
    global show_imgs_n_centers_dbg
    if b == 1:
        show_imgs_n_centers_dbg = True
    else:
        show_imgs_n_centers_dbg = False


def clean_inf_n_nan(img, n):
    # set the non-finite values (NaN, inf) to n
    # returns 1 where the img is finite and 0 where it is not
    mask = np.isfinite(img)
    #  where mask puts img, else puts n, so where is finite puts img, else puts n
    return np.where(mask, img, n)

# prompts the user to select a part of the image in which all the calculations will be done
def cut_working_area(clr_img, depth_img):
    global refPt
    if 'refPt' not in globals():
        refPt = [0]
    if not len(refPt) == 2:
        print("Selecting working region")
        window_name = 'Select Region'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(window_name, clr_img)
        cv2.setMouseCallback(window_name, click_and_crop)
        while not len(refPt) == 2:
            cv2.waitKey(50)
        cv2.destroyWindow(window_name)
    clr_img = clr_img[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    depth_img = depth_img[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
    return clr_img, depth_img


# checks if the images received are actually new images using timestamps
def new_imgs():
    global depth_timestamp, clr_timestamp, last_clr_timestamp, last_depth_timestamp, depth_img_avg
    if not depth_timestamp == 0 and not clr_timestamp == 0 and not clr_timestamp == last_clr_timestamp and not \
                    depth_timestamp == last_depth_timestamp:
        last_clr_timestamp = clr_timestamp
        last_depth_timestamp = depth_timestamp
        return True
    else:
        return False

# shows depth and color images if needed
def show_imgs():
    if show_color:
        cv2.imshow(RGB_WINDOW_NAME, cv2.resize(img_bgr8_clean, (480, 512)))
        cv2.waitKey(1)
    else:
        cv2.destroyWindow(RGB_WINDOW_NAME)
    if show_depth:
        cv2.imshow(DEPTH_WINDOW_NAME, cv2.resize(depth_img_avg, (480, 512)))
        cv2.waitKey(1)
    else:
        cv2.destroyWindow(DEPTH_WINDOW_NAME)

# gets the useful img area as selected by the user, checks if it contains any useful contours (which resembles a cube),
# finds cubes, prerotates them so at least one side is aligned horizontally and then passes it to a function so it can
# be rotated into an appropriate position
def begin_treatment():
    treatment_lock.acquire()
    show_imgs()
    global depth_img_avg, img_bgr8_clean, show_imgs_n_centers_dbg
    if not new_imgs():
        treatment_lock.release()
        return
    # work only on the area the user selected
    clr, dpht = cut_working_area(img_bgr8_clean.copy(), depth_img_avg.copy())
    cv2.imshow('Working Area', clr)
    cv2.waitKey(1)
    # find the appropriate contours (which resembles a cube)
    useful_cnts = find_cnts_in_depth(dpht)
    if useful_cnts is None:
        treatment_lock.release()
        return
    imgs_n_centers = prerotate_cubes_n_get_center(useful_cnts, clr.copy(), False)
    if imgs_n_centers is None:
        treatment_lock.release()
        return
    # debugging code
    if 'show_imgs_n_centers_dbg' in globals():
        if show_imgs_n_centers_dbg:
            imgs_n_centers_dbg = prerotate_cubes_n_get_center(useful_cnts, clr.copy(), False)
            nb_objs = 0
            for index, tuples in enumerate(imgs_n_centers_dbg):
                img, center = tuples
                cv2.imshow('Objs ' + str(index), img)
                nb_objs = index
            for i in range(nb_objs + 1, nb_objs + 10, 1):
                cv2.destroyWindow('Objs ' + str(i))
            cv2.waitKey(0)
    orientate_imgs(imgs_n_centers)
    treatment_lock.release()

# returns one image from the tuple
def only_get_one_img(imgs_n_centers):
    if imgs_n_centers is None:
        return
    for index, curr_tuple in enumerate(imgs_n_centers):
        img_bgr8, center = curr_tuple
        img_bgr8_copy = img_bgr8.copy()
        return img_bgr8_copy


# returns the HoG image obtained using Sklearns lib
def get_img_hog(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)
    n_bin = 9
    b_size = 8
    c_size = 8
    fd, hog_img = hog(img, orientations=n_bin, pixels_per_cell=(c_size, c_size),
                        cells_per_block=(int(b_size / c_size), int(b_size / c_size)), visualise=True)
    hog_img = exposure.rescale_intensity(hog_img, in_range=(0, 8))
    return fd, hog_img


def orientate_imgs(imgs_n_centers):
    orientation_method = 2  # 0 = HSV + contours, 1 = PCA, 2 = machine learning
    if orientation_method == 0:
        # Using HSV filtering #
        # Does not really work, just to visualize 'why' is does not work and maybe expand it
        min_area_triang.objects_detector(only_get_one_img(imgs_n_centers))
    elif orientation_method == 1:
        # Here is the PCA approach
        # Apply only the PCA rotation (narrows the image to 2 possible position, may be upside down)

        # Only send one image to it since it is not supposed to be used in a real scenario in its current state
        img_180 = pca.apply_pca_rotation(only_get_one_img(imgs_n_centers))
        # resolve the "maybe it's upside" ambiguity using the location of sift features on the image
        img_90 = features_based.sifts_up(img_180, 0)
        # or resolve it using the number of contour points in its up or down side.
        # img_90 = features_based.countourpnts_up(img_180)
        cv2.imshow('RotatedImg', img_90)
        cv2.waitKey(1)
    elif orientation_method == 2:
        # Using machine learning, this is the method that actually works
        final_imgs = machinelearning.objects_detector(imgs_n_centers)
        send_imgs(final_imgs)


# actually sends the images to ROS using the detected objects message, only sends if a speech is detected
# also dumps a file containing sentences describing the current objs being shown (only works with one object at a time)
def send_imgs(img_list):
    global got_speech
    if img_list is None or len(img_list) == 0:
        return
    detected_objects_list = list()
    for index, img in enumerate(img_list):
        cv2.imshow('Img ' + str(index), img)
        cv2.waitKey(1)
        rows, cols, d = img.shape
        detected_object = Detected_Object()
        detected_object.id = index
        detected_object.image = CvBridge().cv2_to_imgmsg(img, encoding="passthrough")
        detected_object.center_x = rows / 2
        detected_object.center_y = cols / 2
        hog, image_hog = get_img_hog(img)
        detected_object.features.hog_histogram = hog
        detected_object.hog_image = CvBridge().cv2_to_imgmsg(image_hog, encoding="passthrough")
        colors_histo, object_shape = getpixelfeatures(img)
        detected_object.features.colors_histogram = colors_histo.tolist()
        # detected_object.features.shape_histogram = object_shape.tolist()
        detected_objects_list.append(detected_object)
        # here dumps the sentences describing the shown object (only works with one object at a time and a known object)
        orientation.automatic_descriptor.automatic_descriptor(orientation.machinelearning.label_pred(img), detected_object)
    if got_speech == 0:
        return
    detected_objects_list_publisher.publish(detected_objects_list)
    got_speech = 0


# Callback from the audio related topic. Receives the current dictionary, words histogram and last spoken words (speech)
# also sets a flag so the program is aware that a valid audio was recorded.



def callback_audio_recognition(words):
    global speech
    global got_speech
    if not words.complete_words:
        return
    speech = words.complete_words
    got_speech = 1


# gets the coordinates of the rectangle indicating the selected working are by the user
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))

# actually filters the contours, checking if they resemble cubes
def filter_cnts(cnt_list):
    global using_VGA
    useful_cnts = list()
    # uses area and length to filter
    for cnt in cnt_list:
        if not using_VGA:
            if MIN_AREA < cv2.contourArea(cnt) < MAX_AREA:
                if MIN_CNT_LENGTH < cv2.arcLength(cnt, 1) < MAX_CNT_LENGTH:
                    useful_cnts.append(cnt)
                else:
                    # print("Wrong Lenght " + str(MIN_CNT_LENGTH) + " < " + str(cv2.arcLength(cnt, 1)) +
                    #       " < " + str(MAX_CNT_LENGTH))
                    continue
            else:
                # print ("Wrong Area: " + str(MIN_AREA) + " < " + str(cv2.contourArea(cnt)) + " < " + str(MAX_AREA))
                continue
        else:
            if MIN_AREA / 2 < cv2.contourArea(cnt) < MAX_AREA / 2:
                if MIN_CNT_LENGTH / 1.4 < cv2.arcLength(cnt, 1) < MAX_CNT_LENGTH / 1.4:
                    useful_cnts.append(cnt)
                else:
                    print("Wrong Lenght " + str(MIN_CNT_LENGTH / 1.4) + " < " + str(cv2.arcLength(cnt, 1)) +
                          " < " + str(MAX_CNT_LENGTH / 1.4))
                    continue
            else:
                print ("Wrong Area: " + str(MIN_AREA / 2) + " < " + str(cv2.contourArea(cnt)) + " < " +
                       str(MAX_AREA / 2))
                continue
    return useful_cnts


# gets cubes contours from useful_cnts, finds a minimal area bounding box and rotates it so a side is aligned
# horizontally
def prerotate_cubes_n_get_center(useful_cnts, clr_copy, draw_cnts_n_box):
    uprightrects_tuples = list()
    # for each contour
    for index, cnts in enumerate(useful_cnts):
        min_area_rect = cv2.minAreaRect(cnts)  # minimum area rectangle that encloses the contour cnt
        (center, size, angle) = cv2.minAreaRect(cnts)   # get the rectangle dimensions
        width, height = size[0], size[1]
        if not (0.65 * height < width < 1.35 * height):  # filter contours which do not resemble rectangles
            print("Wrong Height/Width: " + str(0.65 * height) + " < " + str(width) + " < " + str(1.35 * height))
            continue
        points = cv2.boxPoints(min_area_rect)  # Find four vertices of rectangle from above rect
        points = np.int32(np.around(points))  # Round the values and make it integers
        if draw_cnts_n_box: # draw the bounding box if necessary
            cv2.drawContours(clr_copy, [points], 0, (0, 0, 255), 2)
            cv2.drawContours(clr_copy, cnts, -1, (255, 0, 255), 2)
        # if we the angle goes below -45 rotate it 90 degrees and ajust width and height to reflect it
        # this is because OpenCV only considers values from -90 to 0 for the rotation, so this is a way to
        # keep track of a wider range of rotations
        if angle < -45.0:
            angle += 90.0
            width, height = size[0], size[1]
            size = (height, width)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        # rotate the entire image around the center of the parking cell by the
        # angle of the rotated rect
        # get dimmensions of the image
        imgwidth, imgheight = (clr_copy.shape[0], clr_copy.shape[1])
        # actually rotate it
        rotated = cv2.warpAffine(clr_copy, rot_matrix, (imgheight, imgwidth), flags=cv2.INTER_CUBIC)
        # extract the rectangle after rotation has been done
        sizeint = (np.int32(size[0]), np.int32(size[1]))
        uprightrect = cv2.getRectSubPix(rotated, sizeint, center)
        uprightrect = cv2.resize(uprightrect, (125, 125))
        uprightrects_tuples.append((uprightrect, center))
    if len(uprightrects_tuples) > 0:
        return uprightrects_tuples
    else:
        return None


def find_cnts_in_depth(depth):
    # Uses the depth image to only take the part of the image corresponding to the closest point and a bit further
    # resize the depth image so it matches the color one
    depth_copy = copy.copy(depth)
    closest_pnt = np.amin(depth)
    stable = True
    if stable:
        img_detection = np.where(depth_copy < closest_pnt + val_depth_capture, depth_copy, 0)
        # put all the pixels greater than 0 to 255 to facilitate the contour extraction
        ret, mask = cv2.threshold(img_detection, 0.0, 255, cv2.THRESH_BINARY)
        # convert to 8-bit
        mask = np.array(mask, dtype=np.uint8)
        # actually find the contours
        im2, contours, hierarchy = cv2.findContours(mask, 1, 2, offset=(0, -6)) # offset because the depth image is a
        # a bit offset in respect to the color image
        if len(contours) == 0:
            return None
        # filter contours
        useful_cnts = filter_cnts(contours)
        if len(useful_cnts) == 0:
            return None
        return useful_cnts
    else:
        # here the idea is to use ranges of the depth image instead of just one range as in the method above
        # the ranges are in this form [closest detected point, X], [X-y, X2]...
        # Ex: [0.1, 0.3], [0.2, 0.5]...
        useful_cnts = list()
        overlap = 0.01
        # [0, 0.3
        for lower in np.arange(closest_pnt, closest_pnt + 0.20, 0.03):
            # get all points lower than the starting point (lower) plus a range (val_depth_capture)
            img_detection = np.where(depth_copy < lower + val_depth_capture, depth_copy, 0)
            # get all points greater than lower minus the overlap between ranges
            img_detection = np.where(img_detection > lower - overlap, img_detection, 0)
            # put all the pixels greater than 0 to 255
            ret, mask = cv2.threshold(img_detection, 0.0, 255, cv2.THRESH_BINARY)
            # convert to 8-bit
            mask = np.array(mask, dtype=np.uint8)
            im2, contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE,
                                                        offset=(0, -6))
            if len(contours) == 0:
                continue
            contours = filter_cnts(contours)
            useful_cnts.extend(contours)
    #   checking for nested contours
    # TODO: which contour to get when they are nested ?
    singular_cnts = list()
    if len(useful_cnts) > 0:
        for new_cnts in useful_cnts:    # for all contours
            contours_nested = list()
            found_nested = False
            for cnts in useful_cnts:  # for each point in each contour
                if np.all(new_cnts == cnts):
                    continue
                for useful_points in cnts:
                    if not cv2.pointPolygonTest(new_cnts, (useful_points[0][0], useful_points[0][1]), False) == -1:
                        contours_nested.append(cnts)
                        found_nested = True
                        continue
            if not found_nested:
                singular_cnts.append(new_cnts)

    useful_cnts = singular_cnts
    return useful_cnts


def getpixelfeatures(object_img_bgr8):
    # resize images to the value Yuxin uses in his program
    object_img_bgr8 = cv2.resize(object_img_bgr8, (40, 40))
    # remove the white pixels by filtering all values with saturation lower than 100
    object_img_hsv = cv2.cvtColor(object_img_bgr8, cv2.COLOR_BGR2HSV)
    new_hue = list()
    x, y, d = np.shape(object_img_hsv)
    n_colors = 80
    for i in range(x):
        for i2 in range(y):
            if object_img_hsv[i, i2, 1] > 100:
                new_hue.append(object_img_hsv[i, i2, 0])
    # generate the color histogram
    colors_histo, histo_bins = np.histogram(new_hue, bins=n_colors, range=(0, 179), density=True)
    # code provided by Yuxin, basically smooths the histogram so it resembles a gaussian
    half_segment = int(N_COLORS / 4)
    middle = colors_histo * np.array([0.0] * half_segment + [1.0] * 2 * half_segment + [0.0] * half_segment)
    sigma = 2.0
    middle = ndimage.filters.gaussian_filter1d(middle, sigma)
    exterior = colors_histo * np.array([1.0] * half_segment + [0.0] * 2 * half_segment + [1.0] * half_segment)
    exterior = np.append(exterior[2 * half_segment:], exterior[0:2 * half_segment])
    exterior = ndimage.filters.gaussian_filter1d(exterior, sigma)
    colors_histo = middle + np.append(exterior[2 * half_segment:], exterior[0:2 * half_segment])
    object_shape = cv2.cvtColor(object_img_bgr8, cv2.COLOR_BGR2GRAY)
    object_shape = np.ndarray.flatten(object_shape)
    sum_shape = float(np.sum(object_shape))
    if sum_shape != 0:
        # wanted to replace with augmented assignment, but the float conversion has unexpected results in some python
        # versions if that is done
        object_shape = object_shape/float(np.sum(object_shape))
    features = Vision_Features()
    features.colors_histogram = colors_histo
    features.shape_histogram = object_shape
    return colors_histo, object_shape

if __name__ == '__main__':
    rospy.init_node('imageToObjects', anonymous=True)
    print("Creating windows")
    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Show Color Image', MAIN_WINDOW_NAME, 0, 1, show_clr_imgs)
    cv2.createTrackbar('Show Depth Image', MAIN_WINDOW_NAME, 0, 1, show_depth_imgs)
    cv2.createTrackbar('Show Objs', MAIN_WINDOW_NAME, 0, 1, show_objs)
    print("Creating subscribers")
    image_sub_rgb = rospy.Subscriber("/camera/rgb/image_raw", Image, callback_rgb, queue_size=1)
    image_sub_depth = rospy.Subscriber("/camera/depth/image/", Image, callback_depth, queue_size=1)
    detected_objects_list_publisher = rospy.Publisher('detected_objects_list', Detected_Objects_List, queue_size=1)
    object_sub = rospy.Subscriber("audition_features", Audition_Features, callback_audio_recognition)
    print("Spinning ROS")
    try:
        while not rospy.core.is_shutdown():
            begin_treatment()
            # prevents some windows from not showing if waitKey is forgot
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("Shutting down")
        cv2.destroyAllWindows()
        exit(1)
