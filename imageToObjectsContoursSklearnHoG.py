#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import rospy
import numpy as np
from scipy import ndimage
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from robot_interaction_experiment.msg import Vision_Features
from robot_interaction_experiment.msg import Detected_Object
from robot_interaction_experiment.msg import Detected_Objects_List
import scipy
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn import neighbors
from skimage.feature import hog
from skimage import exposure
import os
import cv2
from sklearn.neural_network import BernoulliRBM
import pickle
from sklearn.externals import joblib
import copy
import time

WIDTH = 1280
HEIGHT = 960
MAIN_WINDOW_NAME = "Segmentator"
MIN_AREA = 9000
MAX_AREA = 15000
N_COLORS = 80
TRACKBAR_NB_PROFONDEUR_NAME = "Nb img profondeur"
NB_IMG_PROFONDEUR_MAX = 100
NB_DEPTH_IMGS = 30
AFFICHAGE_PROFONDEUR = "Afficher img profondeur"
show_depth = False
AFFICHAGE_COULEUR = "Afficher img couleur"
show_color = False
CAPTURE_PROFONDEUR = "capture profondeur"
val_depth_capture = 0.04
interactive = 0
depthImgIndex = 0
pointsIndex = 0
NB_INDEX_BOX = 7
lastDepthImgs = range(NB_IMG_PROFONDEUR_MAX + 1)
NUMBER_LAST_POINTS = 15
lastpoints = np.zeros((NUMBER_LAST_POINTS, 3, 2))
lastBoxes = range(NB_INDEX_BOX + 1)
hog_list = list()
last_hog = 0
indiceboxes = 0
depth_img_Avg = 0
img_bgr8_clean = 0
got_color = False
got_depth = False
average_points = np.zeros((3, 2))
clf = svm.SVC(probability=True)
labels = list()
recording = 0
label = ''
INTERACTIONS = 0
loaded = 0
DEBUG = 0
show = 0
saving_learn = 0
saved = 0
color = ''
n_bin = 3  # number of orientations for the HoG
b_size = 12  # block size
c_size = 12  # cell size
rotation = -1
saving_test = 0
failure = 0
total = 0
tst_dsk_percentage = -1
RECORDING = 0
SHOW = 0
SAVING_LEARN = 0
SAVED = 0
SAVING_TEST = 0
def nothing(x):
    pass

def save_imgs_learn(value):
    mode = str(raw_input('Label: '))
    global label
    LABEL = mode
    color_ = str(raw_input('Color: '))
    global color
    COLOR = color_
    global saving_learn
    SAVING_LEARN = 1


def save_imgs_test(value):
    mode = str(raw_input('Label: '))
    global label
    LABEL = mode
    color_ = str(raw_input('Color: '))
    global color
    COLOR = color_
    global rotation
    ROTATION = str(raw_input('Rotation: '))
    global saving_test
    SAVING_TEST = 1

def changecapture(n):
    global val_depth_capture
    if n == 0:
        n = 1
    VAL_DEPTH_CAPTURE = float(n) / 100


def changeprofondeur(n):
    global NB_DEPTH_IMGS
    NB_DEPTH_IMGS = n
    if NB_DEPTH_IMGS <= 0:
        NB_DEPTH_IMGS = 1


def changeaffprofondeur(b):
    global show_depth
    if b == 1:
        SHOW_DEPTH = True
    else:
        SHOW_DEPTH = False


def changeaffcouleur(b):
    global show_color
    if b == 1:
        SHOW_COLOR = True
    else:
        SHOW_COLOR = False


def clean(img, n):
    # set the non-finite values (NaN, inf) to n
    # returns 1 where the img is finite and 0 where it is not
    mask = np.isfinite(img)
    #  where mask puts img, else puts n, so where is finite puts img, else puts n
    return np.where(mask, img, n)


def callback_depth(msg):
    # treating the image containing the depth data
    global depthImgIndex, lastDepthImgs, depth_img_Avg, got_depth
    # getting the image
    try:
        img = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    except CvBridgeError as e:
        print (e)
        return
    cleanimage = clean(img, 255)
    if show_depth:
        # shows the image after processing
        cv2.imshow("Depth", img)
        cv2.waitKey(1)
    else:
        cv2.destroyWindow("Depth")
    # storing the image
    lastDepthImgs[depthImgIndex] = np.copy(cleanimage)
    depthImgIndex += 1
    if depthImgIndex > NB_DEPTH_IMGS:
        depthImgIndex = 0
    # creates an image which is the average of the last ones
    depth_img_Avg = np.copy(lastDepthImgs[0])
    for i in range(0, NB_DEPTH_IMGS):
        depth_img_Avg += lastDepthImgs[i]
    depth_img_Avg /= NB_DEPTH_IMGS
    got_depth = True  # ensures there is an depth image available
    if got_color and got_depth:
        filter_by_depth()



def callback_rgb(msg):
    # processing of the color image
    global img_bgr8_clean, got_color
    # getting image
    try:
        img = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError as e:
        print (e)
        return

    # img = cv2.resize(img, (WIDTH, HEIGHT))
    # print np.shape(img)

    img = img[32:992, 0:1280]  # crop the image because it does not have the same aspect ratio of the depth one
    img_bgr8_clean = np.copy(img)

    got_color = True  # ensures there is an color image available
    if show_color:
        # show image obtained
        cv2.imshow("couleur", img_bgr8_clean)
        cv2.waitKey(1)
    else:
        cv2.destroyWindow("couleur")


def filter_by_depth():
    # Uses the depth image to only take the part of the image corresponding to the closest point and a bit further
    global depth_img_Avg
    closest_pnt = np.amin(depth_img_Avg)
    # print np.shape(depth_img_Avg)
    depth_img_Avg = cv2.resize(depth_img_Avg, (WIDTH, HEIGHT))
    # print np.shape(depth_img_Avg)

    # generate a mask with the closest points
    img_detection = np.where(depth_img_Avg < closest_pnt + val_depth_capture, depth_img_Avg, 0)
    # put all the pixels greater than 0 to 255
    ret, mask = cv2.threshold(img_detection, 0.0, 255, cv2.THRESH_BINARY)
    mask = np.array(mask, dtype=np.uint8)  # convert to 8-bit
    im2, contours, hierarchy = cv2.findContours(mask, 1, 2, offset=(0, -6))
    biggest_cont = contours[0]
    for cnt in contours:
        if cv2.contourArea(cnt) > cv2.contourArea(biggest_cont):
            biggest_cont = cnt
    min_area_rect = cv2.minAreaRect(biggest_cont)  # minimum area rectangle that encloses the contour cnt
    (center, size, angle) = cv2.minAreaRect(biggest_cont)
    points = cv2.boxPoints(min_area_rect)  # Find four vertices of rectangle from above rect
    points = np.int32(np.around(points))  # Round the values and make it integers
    img_bgr8_clean_copy = img_bgr8_clean.copy()
    cv2.drawContours(img_bgr8_clean_copy, [points], 0, (0, 0, 255), 2)
    cv2.drawContours(img_bgr8_clean_copy, biggest_cont, -1, (255, 0, 255), 2)
    cv2.imshow('RBG', img_bgr8_clean_copy)
    cv2.waitKey(1)

    # if we rotate more than 90 degrees, the width becomes height and vice-versa
    if angle < -45.0:
        angle += 90.0
        width, height = size[0], size[1]
        size = (height, width)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotate the entire image around the center of the parking cell by the
    # angle of the rotated rect
    imgwidth, imgheight = (img_bgr8_clean.shape[0], img_bgr8_clean.shape[1])
    rotated = cv2.warpAffine(img_bgr8_clean, rot_matrix, (imgheight, imgwidth), flags=cv2.INTER_CUBIC)
    # extract the rect after rotation has been done
    sizeint = (np.int32(size[0]), np.int32(size[1]))
    uprightrect = cv2.getRectSubPix(rotated, sizeint, center)
    uprightrect_copy = uprightrect.copy()
    cv2.drawContours(uprightrect_copy, [points], 0, (0, 0, 255), 2)
    cv2.imshow('uprightRect', uprightrect_copy)

    objects_detector(uprightrect)

def hog_pred(value):
    global n_bin
    global b_size
    global c_size
    global img_clean_GRAY_class
    fd = hog(img_clean_GRAY_class, orientations=n_bin, pixels_per_cell=(c_size, c_size),
                        cells_per_block=(b_size / c_size, b_size / c_size), visualise=False)
    global clf
    print (clf.predict_proba([fd]))
    print (clf.predict([fd]))

def load_class(value):
    global clf
    global hog_list
    global labels
    clf = joblib.load('Classifier/filename.pkl')
    with open('HOG_N_LABELS/HOG_N_LABELS.pickle') as f:
        HOG_TUPLE = pickle.load(f)
    HOG_LIST = HOG_TUPLE[0]
    labels = HOG_TUPLE[1]
    print (clf)
    global loaded
    LOADED = 1
    print ('Loaded')

def show(value):
    global show
    SHOW = value

def debug(value):
    global DEBUG
    DEBUG = value

def learn(value):
    clf.fit(hog_list, labels)
    print ('Done')

def save_class(value):
    global clf
    HOG_TUPLE = (hog_list, labels)
    print ('Hog = ' + str(HOG_TUPLE[0]))
    print ('labels = ' + str(HOG_TUPLE[1]))
    clf.fit(HOG_TUPLE[0], HOG_TUPLE[1])
    with open('HOG_N_LABELS/HOG_N_LABELS.pickle', 'w') as f:
        pickle.dump(HOG_TUPLE, f)
    joblib.dump(clf, 'Classifier/filename.pkl')
    print ('Done')


def hog_appender(value):
    global recording
    global last_hog
    global hog_list
    global clf
    global labels
    print ('Already have these labels:')
    myset = set(labels)
    print (str(myset))
    mode = str(raw_input('Label: '))
    global label
    LABEL = mode
    RECORDING = 1

def hog_info(value):
    global labels
    global hog_list
    print ('Current labels = ')
    myset = set(labels)
    print (str(myset))
    print ('Current HoG size:')
    print (len(HOG_LIST))


def objects_detector(img_bgr8):
    global RECORDING
    global SHOW
    global SAVING_LEARN
    global SAVED
    global SAVING_TEST
    width, height, d = np.shape(img_bgr8)
    if width > 130 or height > 130:
        return
    if width < 100 or height < 100:
        return
    detected_objects_list = []
    w, l, d = np.shape(img_bgr8)
    global img_clean_BGR_learn
    img_clean_BGR_learn = img_bgr8[2:w-2, 2:l-2].copy()
    cv2.imshow('Learn',img_clean_BGR_learn)
    img_bgr8 = img_bgr8[7:w-4, 9:l-8]
    img_clean_BGR_class = img_bgr8.copy()
    img_clean_BGR_class = cv2.resize(img_clean_BGR_class, (115, 120), interpolation=cv2.INTER_AREA)  # resize image
    global img_clean_GRAY_class
    img_clean_GRAY_class = cv2.cvtColor(img_clean_BGR_class, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Clean', img_clean_BGR_class)
    img_bgr8_copy = img_bgr8.copy()
    hsv = cv2.cvtColor(img_bgr8_copy, cv2.COLOR_RGB2HSV)
    # define the values range
    hh = 255
    hl = 0
    sh = 255
    sl = 40  # filter the white color background
    vh = 255
    vl = 0  # to ignore the black in the background
    lowerbound = np.array([hl, sl, vl], np.uint8)
    upperbound = np.array([hh, sh, vh], np.uint8)
    # filter the image to generate the mask
    filtered_hsv = cv2.inRange(hsv, lowerbound, upperbound)
    filtered_hsv = cv2.bitwise_and(hsv, hsv, mask=filtered_hsv)
    cv2.imshow('Filtered', filtered_hsv)
    cv2.waitKey(1)
    # convert the image to grayscale in order to find contours
    img_bgr = cv2.cvtColor(filtered_hsv, cv2.COLOR_HSV2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray,3)
    # img_gray = cv2.bilateralFilter(img_gray, 25, 4, 20)
    # ret, img_gray = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow('Filtered grayscale', img_gray)
    cv2.waitKey(1)
    img_gray_copy = img_gray.copy()
    im2, contours, hierarchy = cv2.findContours(img_gray_copy.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Find the index of the largest contour
    if not contours:
        print ('No contours found =(')
        return
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    # epsilon = 0.005 * cv2.arcLength(cnt, True)
    # cnt = cv2.approxPolyDP(cnt, epsilon, True)
    height, width, channels = img_bgr8_copy.shape
    contour_img = img_bgr8_copy.copy()
    cv2.drawContours(contour_img, cnt, -1, (0, 255, 0), 3)
    x, y, width, height = cv2.boundingRect(cnt)
    contour_img_box = contour_img.copy()
    cv2.imshow('Contour', cv2.resize(contour_img_box, (256, 256)))
    cv2.waitKey(1)
    cropped_bgr8 = img_bgr8_copy[y:y + height, x:x + width]
    cv2.imshow('zica', cropped_bgr8)
    cropped_gray = cv2.cvtColor(cropped_bgr8, cv2.COLOR_BGR2GRAY)
    cropped_gray = cv2.resize(cropped_gray, (100, 100), interpolation=cv2.INTER_AREA)  # resize image
    cv2.imshow('CropGray', cropped_gray)
    fd, hog_image = hog(img_clean_GRAY_class, orientations=n_bin, pixels_per_cell=(c_size, c_size),
                        cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 4))
    cv2.imshow('HOGG', hog_image)
    global clf
    global last_hog
    global n_bin
    global b_size
    global c_size
    global saving_learn
    global recording
    if RECORDING == 1:
        global INTERACTIONS
        learn_hog(img_clean_BGR_learn)
        # HOG_LIST.append(last_hog)
        # labels.append(LABEL)
        INTERACTIONS += 1
        print (INTERACTIONS)
        if INTERACTIONS == 20:
            RECORDING = 0
            INTERACTIONS = 0
            print ('Done recording')
    global saved
    global saving_learn
    if SAVING_LEARN == 1:
        cv2.imwrite('LRN_IMGS/' + label + '_' + str(SAVED) + '_' + color + '.png', img_clean_BGR_learn)
        SAVED += 1
        print (SAVED)
        if SAVED == 20:
            SAVING_LEARN = 0
            SAVED = 0
            print ('Done saving')
    global saving_test
    cv2.imshow('Save_test', img_clean_BGR_class)
    if SAVING_TEST == 1:
        cv2.imwrite('TST_IMGS/' + label + '_' + str(rotation) + '_' +
                    str(SAVED) + '_' + color + '.png', img_clean_BGR_class)
        SAVED += 1
        print (SAVED)
        if SAVED == 20:
            SAVING_TEST = 0
            SAVED = 0
            print ('Done saving')
    best_rot = 0
    best_perc = 0
    global show
    if SHOW == 0:
        return
    if not hasattr(clf, 'support_'):
        return

    global DEBUG
    for i in range(4):
        fd, hog_image = hog(img_clean_GRAY_class, orientations=n_bin, pixels_per_cell=(c_size, c_size),
                            cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)
        for percentage in clf.predict_proba(fd)[0]:
            if percentage > best_perc:
                best_perc = percentage
                best_rot = i
        rows, cols = img_clean_GRAY_class.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img_clean_GRAY_class = cv2.warpAffine(img_clean_GRAY_class, M, (cols, rows))
    if DEBUG == 1:
        # print clf.predict(fd)
        print (best_perc)
        print ('\n')
    print (best_rot)
    if not best_rot == 0:
        rows, cols, d = img_clean_BGR_class.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), best_rot*90, 1)
        img_clean_BGR_class = cv2.warpAffine(img_clean_BGR_class, M, (cols, rows))
    cv2.imshow('Sent', cv2.resize(img_clean_BGR_class, (256, 256)))

    #     detected_object = Detected_Object()
    #     detected_object.id = count
    #     detected_object.image = CvBridge().cv2_to_imgmsg(img_bgr8_resized, encoding="passthrough")
    #     detected_object.center_x = unrot_center_x / float(resolution_x)  # proportion de la largeur
    #     detected_object.center_y = unrot_center_y / float(resolution_x)  # proportion de la largeur aussi
    #     detected_object.features = getpixelfeatures(object_img_rgb)
    #     detected_object.features.hog_histogram = GetHOGFeatures(object_img_rgb)
    #     detected_objects_list.append(detected_object)
    # if interactive == 1:
    #     if len(detected_objects_list) > 1:
    #         VAL_DEPTH_CAPTURE -= 0.01
    #     if len(detected_objects_list) < 1:
    #         VAL_DEPTH_CAPTURE += 0.01
    # detected_objects_list_msg = Detected_Objects_List()
    # detected_objects_list_msg.detected_objects_list = detected_objects_list
    # detected_objects_list_publisher.publish(detected_objects_list_msg)

    # cv2.rectangle(img_copy, (margin, margin), (resolution_x - margin, resolution_y - margin), (255, 255, 255))
    # cv2.imshow('detected_object', img_copy)
    # try:
    #     img_bgr8_resized
    # except NameError:
    #     pass
    # else:
    #     if 1:
    #         cv2.imshow('a', img_bgr8_resized)
    #         cv2.imshow('ROTATED', rotated_img_obj)
    #         cv2.imshow('With Cnt', object_img_rgb2)
    #     cv2.waitKey(1)

def get_img_rot(img_bgr):
    img_clean_GRAY_class_local = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    best_rot = 0
    best_perc = 0
    for i in range(4):
        fd, hog_image = hog(img_clean_GRAY_class_local, orientations=n_bin, pixels_per_cell=(c_size, c_size),
                            cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)
        for percentage in clf.predict_proba(fd)[0]:
            if percentage > best_perc:
                best_perc = percentage
                best_rot = i
        rows, cols = img_clean_GRAY_class_local.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img_clean_GRAY_class_local = cv2.warpAffine(img_clean_GRAY_class_local, M, (cols, rows))
    return best_rot



def learn_hog(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global n_bin
    global b_size
    global c_size
    global hog_list
    global labels
    w, l = np.shape(img)
    img_list = list()
    img_list.append(img)
    img_list.append(img[5:w - 2, 7:l - 6])
    img_list.append(img[4:, :])
    img_list.append(img[0:w-4, :])
    img_list.append(img[:, 4:])
    img_list.append(img[:, :l-4])
    img_list.append(img[7:, :])
    img_list.append(img[0:w-7, :])
    img_list.append(img[:, 7:])
    img_list.append(img[:, :l-7])
    img_list.append(img[12:, :])
    img_list.append(img[0:w-12, :])
    img_list.append(img[:, 12:])
    img_list.append(img[:, :l-12])
    index = 0
    global show
    for imgs in img_list:
        imgs = cv2.resize(imgs, (115, 120), interpolation=cv2.INTER_AREA)  # resize image
        if SHOW == 1:
            cv2.imshow('img' + str(index), imgs)
        index += 1
        HOG_LIST.append(hog(imgs, orientations=n_bin, pixels_per_cell=(c_size, c_size),
                            cells_per_block=(b_size / c_size, b_size / c_size), visualise=False))
        labels.append(label)


def test_from_disk(value):
    path = 'TST_IMGS/'
    global label
    global rotation
    global total
    global tst_dsk_percentage
    total = 0
    global failure
    failure = 0
    for filename in os.listdir(path):
        total += 1
        LABEL = filename.rsplit('_', 3)[0]
        ROTATION = int(filename.rsplit('_', 3)[1])
        # print 'Label ' + str(LABEL)
        # print 'Rotation ' + str(ROTATION)
        imagee = cv2.imread(path + filename)
        found_rot = get_img_rot(imagee)
        if not abs(ROTATION - found_rot) < 0.5:
            print ('Testing ' + str(filename))
            failure += 1
            print ('Does not work')
            # cv2.imshow('Did not work',imagee)
            # cv2.waitKey(100)
            print (found_rot)
            print (ROTATION)
    percentage = 100 * failure / total
    print ('Failure = ' + str(percentage) + '%')
    print ('Done')


def learn_from_disk(value):
    path = 'LRN_IMGS/'
    global label
    for filename in os.listdir(path):
        # print 'Learning ' + str(filename)
        LABEL = filename.rsplit('_', 2)[0]
        # print 'Label = ' + str(LABEL)
        imagee = cv2.imread(path + filename)
        learn_hog(imagee)
    learn(1)
    print ('Done')


def big_test(value):
    global n_bin
    global b_size
    global c_size
    for bin_ in range(2, 15, 1):
        for b in range(30, 4, -1):
            start_time = time.time()
            global labels
            global hog_list
            labels = list()
            HOG_LIST = list()
            n_bin = bin_
            b_size = b
            c_size = b
            path = 'LRN_IMGS/'
            global label
            global failure
            global total
            global tst_dsk_percentage
            print ('Creating HoG')
            for filename in os.listdir(path):
                # print 'Learning ' + str(filename)
                LABEL = filename.rsplit('_', 2)[0]
                # print 'Label = ' + str(LABEL)
                imagee = cv2.imread(path + filename)
                learn_hog(imagee)
            print ('Learning HoG')
            learn(1)
            print ('Testing HoG')
            test_from_disk(1)
            print ('Done, writting to file')
            with open('somefile.txt', 'a') as the_file:
                the_file.write('n_bin = ' + str(n_bin) + '\n')
                the_file.write('b_size = ' + str(b_size) + '\n')
                the_file.write('c_size = ' + str(c_size) + '\n')
                the_file.write('Failure = ' + str(failure) + '\n')
                the_file.write('Total = ' + str(total) + '\n')
                the_file.write('Percentage = ' + str(percentage) + '\n')
                the_file.write('Elapsed Time = ' + str(time.time() - start_time) + '\n\n\n')
            print('Written')
            print ('Elapsed Time = ' + str(time.time() - start_time) + '\n')
    print ('Big Test Done')


def GetHOGFeatures(object_img_rgb):
    std_length = 80
    global n_bin
    global b_size
    global c_size
    h, w, z = np.shape(object_img_rgb)
    img = cv2.resize(object_img_rgb, (std_length, std_length), interpolation=cv2.INTER_AREA)
    img = img[:, :, 1]
    fd, hog_image = hog(img, orientations=n_bin, pixels_per_cell=(c_size, c_size),
                        cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 4))
    cv2.imshow("IMAGE OF HIST", hog_image)
    return fd


def getpixelfeatures(object_img_bgr8):
    object_img_hsv = cv2.cvtColor(object_img_bgr8, cv2.COLOR_BGR2HSV)
    # gets the color histogram divided in N_COLORS "categories" with range 0-179
    colors_histo, histo_bins = np.histogram(object_img_hsv[:, :, 0], bins=N_COLORS, range=(0, 179))
    colors_histo[0] -= len(np.where(object_img_hsv[:, :, 1] == 0)[0])
    half_segment = N_COLORS / 4
    middle = colors_histo * np.array([0.0] * half_segment + [1.0] * 2 * half_segment + [0.0] * half_segment)
    sigma = 2.0
    middle = ndimage.filters.gaussian_filter1d(middle, sigma)
    exterior = colors_histo * np.array([1.0] * half_segment + [0.0] * 2 * half_segment + [1.0] * half_segment)
    exterior = np.append(exterior[2 * half_segment:], exterior[0:2 * half_segment])
    exterior = ndimage.filters.gaussian_filter1d(exterior, sigma)
    colors_histo = middle + np.append(exterior[2 * half_segment:], exterior[0:2 * half_segment])
    colors_histo = colors_histo / float(np.sum(colors_histo))
    object_shape = cv2.cvtColor(object_img_bgr8, cv2.COLOR_BGR2GRAY)
    object_shape = np.ndarray.flatten(object_shape)
    sum = float(np.sum(object_shape))
    if sum != 0:
        object_shape = object_shape / float(np.sum(object_shape))
    features = Vision_Features()
    features.colors_histogram = colors_histo
    features.shape_histogram = object_shape
    return features


if __name__ == '__main__':
    rospy.init_node('imageToObjects', anonymous=True)
    print ("Creating windows")
    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(TRACKBAR_NB_PROFONDEUR_NAME, MAIN_WINDOW_NAME, NB_DEPTH_IMGS, NB_IMG_PROFONDEUR_MAX,
                       changeprofondeur)
    cv2.createTrackbar('Show', MAIN_WINDOW_NAME, 0, 1, show)
    cv2.createTrackbar(AFFICHAGE_COULEUR, MAIN_WINDOW_NAME, 0, 1, changeaffcouleur)
    cv2.createTrackbar('Learn from DISK', MAIN_WINDOW_NAME, 0, 1, learn_from_disk)
    cv2.createTrackbar('Test from DISK', MAIN_WINDOW_NAME, 0, 1, test_from_disk)
    cv2.createTrackbar('Record HoG', MAIN_WINDOW_NAME, 0, 1, hog_appender)
    cv2.createTrackbar('Save IMGs Learn', MAIN_WINDOW_NAME, 0, 1, save_imgs_learn)
    cv2.createTrackbar('Save IMGs Test', MAIN_WINDOW_NAME, 0, 1, save_imgs_test)
    cv2.createTrackbar('Info HoG', MAIN_WINDOW_NAME, 0, 1, hog_info)
    cv2.createTrackbar('Save Class', MAIN_WINDOW_NAME, 0, 1, save_class)
    cv2.createTrackbar('Big Test', MAIN_WINDOW_NAME, 0, 1, big_test)
    cv2.createTrackbar('Learn', MAIN_WINDOW_NAME, 0, 1, learn)
    cv2.createTrackbar('Load Class', MAIN_WINDOW_NAME, 0, 1, load_class)
    cv2.createTrackbar('Debug', MAIN_WINDOW_NAME, 0, 1, debug)
    cv2.createTrackbar('Predict HoG', MAIN_WINDOW_NAME, 0, 1, hog_pred)
    cv2.createTrackbar(AFFICHAGE_PROFONDEUR, MAIN_WINDOW_NAME, 0, 1, changeaffprofondeur)
    cv2.createTrackbar(CAPTURE_PROFONDEUR, MAIN_WINDOW_NAME, int(100 * val_depth_capture), 150, changecapture)
    cv2.imshow(MAIN_WINDOW_NAME, 0)
    print ("Creating subscribers")
    image_sub_rgb = rospy.Subscriber("/camera/rgb/image_rect_color", Image, callback_rgb, queue_size=1)
    image_sub_depth = rospy.Subscriber("/camera/depth_registered/image_raw/", Image, callback_depth, queue_size=1)
    detected_objects_list_publisher = rospy.Publisher('detected_objects_list', Detected_Objects_List, queue_size=1)
    print ("Spinning ROS")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down")
        cv2.destroyAllWindows()
