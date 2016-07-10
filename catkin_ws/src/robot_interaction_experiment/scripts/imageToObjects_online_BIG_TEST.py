#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import rospy
import numpy as np
from scipy import ndimage
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from robot_interaction_experiment.msg import Vision_Features
from robot_interaction_experiment.msg import Detected_Objects_List
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
import os
import cv2
import pickle
import random
import time
import tsne
MAIN_WINDOW_NAME = "Segmentator"
MIN_AREA = 9000  # minimal area to consider a desirable object
MAX_AREA = 15000  # maximal area to consider a desirable object
N_COLORS = 80  # number of colors to consider when creating the image descriptor
IMG_DEPTH_MAX = 100  # maximum number of depth images to cache
nb_depth_imgs_cache = 30  # default number of depth images to consider
show_depth = False
show_color = False
val_depth_capture = 0.03
depth_img_index = 0
last_depth_imgs = range(IMG_DEPTH_MAX + 1)
NUMBER_LAST_POINTS = 15
hog_list = list()
depth_img_avg = 0
img_bgr8_clean = 0
got_color = False
got_depth = False
labels = list()
recording = 0
label = ''
recording_index = 0
loaded = 0
DEBUG = 0
show = 0
saving_learn = 0
saved = 0
color = ''
n_bin = 6  # 4 number of orientations for the HoG
b_size = 64  # 15  block size
b_stride = 32
c_size = 32  # 15  cell size
rotation = -1
saving_test = 0
failure = 0
total = 0
tst_dsk_percentage = -1
live = 0
shuffled_y = list()
shuffled_x = list()
live_cnt = 0
# CLF = SGDClassifier(loss='log')
IMPLEMENTS_P_FIT = 0
NM_NEIGHBORS = -1


def save_imgs_learn(value):
    global label
    global color
    global saving_learn
    mode = str(raw_input('Label: '))
    LABEL = mode
    color_ = str(raw_input('Color: '))
    COLOR = color_
    SAVING_LEARN = 1


def save_imgs_test(value):
    global label
    global color
    global rotation
    global saving_test
    mode = str(raw_input('Label: '))
    LABEL = mode
    color_ = str(raw_input('Color: '))
    COLOR = color_
    ROTATION = str(raw_input('Rotation: '))
    SAVING_TEST = 1


def changecapture(n):
    global val_depth_capture
    if n == 0:
        n = 1
    VAL_DEPTH_CAPTURE = float(n) / 100


def changeprofondeur(n):
    global nb_depth_imgs_cache
    NB_DEPTH_IMGS_INITIAL = n
    if NB_DEPTH_IMGS_INITIAL <= 0:
        NB_DEPTH_IMGS_INITIAL = 1


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
    global depth_img_index, last_depth_imgs, depth_img_avg, got_depth
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
    LAST_DEPTH_IMGS[DEPTH_IMG_INDEX] = np.copy(cleanimage)
    DEPTH_IMG_INDEX += 1
    if DEPTH_IMG_INDEX > nb_depth_imgs_cache:
        DEPTH_IMG_INDEX = 0
    # creates an image which is the average of the last ones
    DEPTH_IMG_AVG = np.copy(LAST_DEPTH_IMGS[0])
    for i in range(0, nb_depth_imgs_cache):
        DEPTH_IMG_AVG += LAST_DEPTH_IMGS[i]
    DEPTH_IMG_AVG /= nb_depth_imgs_cache
    GOT_DEPTH = True  # ensures there is an depth image available
    if got_color and GOT_DEPTH:
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
    img = img[32:992, 0:1280]  # crop the image because it does not have the same aspect ratio of the depth one
    IMG_BGR8_CLEAN = np.copy(img)
    GOT_COLOR = True  # ensures there is an color image available
    if show_color:
        # show image obtained
        cv2.imshow("couleur", IMG_BGR8_CLEAN)
        cv2.waitKey(1)
    else:
        cv2.destroyWindow("couleur")


def filter_by_depth():
    # Uses the depth image to only take the part of the image corresponding to the closest point and a bit further
    global depth_img_avg
    closest_pnt = np.amin(DEPTH_IMG_AVG)
    # print np.shape(depth_img_Avg)
    DEPTH_IMG_AVG = cv2.resize(DEPTH_IMG_AVG, (1280, 960))
    # print np.shape(depth_img_Avg)
    # generate a mask with the closest points
    img_detection = np.where(DEPTH_IMG_AVG < closest_pnt + val_depth_capture, DEPTH_IMG_AVG, 0)
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
    fd = hog(img_clean_GRAY_class, orientations=N_BIN, pixels_per_cell=(C_SIZE, C_SIZE),
             cells_per_block=(B_SIZE / C_SIZE, B_SIZE / C_SIZE), visualise=False)
    global CLF
    print (CLF.predict([fd]))


def load_class(value):
    global CLF
    global hog_list
    global labels
    with open('HOG_N_LABELS/HOG_N_LABELS.pickle') as f:
        hog_tuple = pickle.load(f)
    HOG_LIST = hog_tuple[0]
    LABELS = hog_tuple[1]
    # print (clf)
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
    print ('Learning')
    start_time = time.time()
    global hog_list
    classes = np.unique(labels).tolist()
    for i in range(10):
        classes.append('new' + str(i))
    print (classes)
    global shuffled_x
    global shuffled_y
    shuffledrange = range(len(labels))
    if IMPLEMENTS_P_FIT == 1:
        for i in range(5):
            random.shuffle(shuffledrange)
            SHUFFLED_X = [HOG_LIST[i] for i in shuffledrange]
            SHUFFLED_Y = [labels[i] for i in shuffledrange]
            print (len(SHUFFLED_X))
            for i2 in range(10):
                print (i2)
                CLF.partial_fit(SHUFFLED_X[i2 * len(SHUFFLED_X) / 10:(i2 + 1) * len(SHUFFLED_X) / 10], SHUFFLED_Y[i2 * len(SHUFFLED_X) / 10:(i2 + 1) * len(SHUFFLED_X) / 10], classes)
    else:
        shuffledrange = range(len(labels))
        random.shuffle(shuffledrange)
        SHUFFLED_X = [HOG_LIST[i] for i in shuffledrange]
        SHUFFLED_Y = [labels[i] for i in shuffledrange]
        CLF.fit(SHUFFLED_X, SHUFFLED_Y)
    print ('Done Learning')
    print('Elapsed Time Learning = ' + str(time.time() - start_time) + '\n')


def save_hog(value):
    global CLF
    HOG_TUPLE = (hog_list, labels)
    # print ('Hog = ' + str(HOG_TUPLE[0]))
    print ('labels = ' + str(np.unique(HOG_TUPLE[1])))
    # clf.fit(HOG_TUPLE[0], HOG_TUPLE[1])
    with open('HOG_N_LABELS/HOG_N_LABELS.pickle', 'w') as f:
        pickle.dump(HOG_TUPLE, f)
    # joblib.dump(clf, 'Classifier/filename.pkl')
    print ('Done')


def hog_info(value):
    global labels
    global hog_list
    print ('Current labels = ')
    myset = set(LABELS)
    print (str(myset))
    print ('Current HoG size:')
    print (len(HOG_LIST))


def objects_detector(img_bgr8):
    objects_detector_time = time.time()
    width, height, d = np.shape(img_bgr8)
    if width > 130 or height > 130:
        return
    if width < 100 or height < 100:
        return
    detected_objects_list = []
    w, l, d = np.shape(img_bgr8)
    global img_clean_BGR_learn
    img_clean_BGR_learn = img_bgr8[2:w-2, 2:l-2].copy()
    cv2.imshow('Learn', img_clean_BGR_learn)
    img_bgr8 = img_bgr8[13:w-5, 13:l-8]
    img_clean_BGR_class = img_bgr8.copy()
    img_clean_BGR_class = cv2.resize(img_clean_BGR_class, (128, 128), interpolation=cv2.INTER_AREA)  # resize image
    global img_clean_GRAY_class
    img_clean_GRAY_class = cv2.cvtColor(img_clean_BGR_class, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Clean', img_clean_BGR_class)
    global CLF
    global n_bin
    global b_size
    global c_size
    global saving_learn
    global recording
    if RECORDING == 1:
        global recording_index
        learn_hog(img_clean_BGR_learn)
        RECORDING_INDEX += 1
        print (RECORDING_INDEX)
        if RECORDING_INDEX == 20:
            RECORDING = 0
            RECORDING_INDEX = 0
            print ('Done recording')
    global saved
    global saving_learn
    if SAVING_LEARN == 1:
        cv2.imwrite('LRN_IMGS/' + label + '_' + str(SAVED) + '_' + color + '.png', img_clean_BGR_learn)
        SAVED += 1
        print (SAVED)
        if SAVED == 3:
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
    global live
    global live_cnt
    global nb_depth_imgs_cache
    global shuffled_x
    global shuffled_y
    global hog_list
    global labels
    global start_time2
    if LIVE == 1:
        start_time3 = time.time()
        if LIVE_CNT == 0:
            start_time2 = time.time()
            LIVE_CNT = 100
            NB_DEPTH_IMGS_INITIAL = 1
            HOG_LIST = list()
            LABELS = list()
            HOG_LIST.extend(SHUFFLED_X[1:int(len(SHUFFLED_X) * (1.0 / len((np.unique(SHUFFLED_Y)))))])
            LABELS.extend(SHUFFLED_Y[1:int(len(SHUFFLED_X) * (1.0 / len(np.unique(SHUFFLED_Y))))])
        learn_hog(img_bgr8)
        shuffledRange = range(len(LABELS))
        for i in range(5):
            random.shuffle(shuffledRange)
            shuffledX_temp = [HOG_LIST[i] for i in shuffledRange]
            shuffledY_temp = [LABELS[i] for i in shuffledRange]
        print (LIVE_CNT)
        LIVE_CNT -= 1
        if LIVE_CNT == 0:
            start_time = time.time()
            CLF.partial_fit(shuffledX_temp, shuffledY_temp)
            print('Elapsed Time LEARNING = ' + str(time.time() - start_time) + '\n')
            LIVE = 0
            NB_DEPTH_IMGS_INITIAL = 30
            print('Elapsed Time TOTAL = ' + str(time.time() - start_time2) + ' FPS = ' + str(100/(time.time() - start_time2)) + '\n')
        print('Elapsed Time Single Example = ' + str(time.time() - start_time3) + '\n')
        print('Elapsed Time Single Example Obj Detc = ' + str(time.time() - objects_detector_time) + '\n')
    global show
    if SHOW == 0:
        return
    global DEBUG
    opencv_hog = cv2.HOGDescriptor((128, 128), (B_SIZE, B_SIZE), (b_stride, b_stride), (C_SIZE, C_SIZE), N_BIN)
    for i in range(4):
        img_clean_GRAY_class = cv2.resize(img_clean_GRAY_class, (128, 128))
        fd4 = opencv_hog.compute(img_clean_GRAY_class)
        fd4 = fd4.reshape(1, -1)
        for pred_percentage in CLF.predict_proba(fd4)[0]:
            if pred_percentage > best_perc:
                best_perc = pred_percentage
                best_rot = i

        rows, cols = img_clean_GRAY_class.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img_clean_GRAY_class = cv2.warpAffine(img_clean_GRAY_class, M, (cols, rows))
    if DEBUG == 1:
        print (best_perc)
        print ('\n')

    # print (best_rot)
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
    img_clean_gray_class_local = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    best_rot = 0
    best_perc = 0
    opencv_hog = cv2.HOGDescriptor((128, 128), (b_size, b_size), (b_stride, b_stride), (c_size, c_size), n_bin)
    for i in range(4):
        h1 = opencv_hog.compute(img_clean_gray_class_local)
        fd = np.reshape(h1, (len(h1),))
        fd = fd.reshape(1, -1)
        for percentage in CLF.predict_proba(fd)[0]:
            if percentage > best_perc:
                best_perc = percentage
                best_rot = i
        rows, cols = img_clean_gray_class_local.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img_clean_gray_class_local = cv2.warpAffine(img_clean_gray_class_local, M, (cols, rows))
    return best_rot


def learn_hog(img):
    start_time = time.time()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    global n_bin
    global b_size
    global c_size
    global hog_list
    global labels
    w, l = np.shape(img)
    img_list = list()
    img_list.append((img[:, :]))  # no changes
    global live
    if LIVE == 0:
        for i in range(1, 12, 2):
            img_list.append((img[0:w-i, :]))  # cut right
            img_list.append((img[i:, :]))  # cut left
            img_list.append((img[:, i:]))  # cut up
            img_list.append((img[:, 0:l-i]))  # cut down
            img_list.append((img[:, i:l-i]))  # cut up and down
            img_list.append((img[i:w-i, :]))  # cut left and right
            img_list.append((img[i:, i:l-i]))  # cut up and down and left
            img_list.append((img[:w-i, i:l-i]))  # cut up and down and right
            img_list.append((img[i:w-i, i:l-i]))  # cut up and down and left and right
    else:
        for i in range(3, 12, 3):
            img_list.append((img[0:w-i, :]))  # cut right
            img_list.append((img[i:, :]))  # cut left
            img_list.append((img[:, i:]))  # cut up
            img_list.append((img[:, 0:l-i]))  # cut down
            img_list.append((img[:, i:l-i]))  # cut up and down
            img_list.append((img[i:w-i, :]))  # cut left and right
            img_list.append((img[i:, i:l-i]))  # cut up and down and left
            img_list.append((img[:w-i, i:l-i]))  # cut up and down and right
            img_list.append((img[i:w-i, i:l-i]))  # cut up and down and left and right
    # hog = cv2.HOGDescriptor()
    index = 0
    global show
    # print('Elapsed Time Pre HoG = ' + str(time.time() - start_time) + '\n')
    start_time = time.time()
    opencv_hog = cv2.HOGDescriptor((128, 128), (B_SIZE, B_SIZE), (b_stride, b_stride), (C_SIZE, C_SIZE), N_BIN)
    for imgs in img_list:
        imgs = cv2.resize(imgs, (128, 128), interpolation=cv2.INTER_AREA)  # resize image
        if SHOW == 1:
            cv2.imshow('img' + str(index), imgs)
        index += 1
        h1 = opencv_hog.compute(imgs)
        h1 = (np.reshape(h1, (len(h1),)))
        HOG_LIST.append((np.reshape(h1, (len(h1),))))
        # print (HOG_LIST)
        # HOG_LIST.append(hog(imgs, orientations=n_bin, pixels_per_cell=(c_size, c_size),
        #                     cells_per_block=(b_size / c_size, b_size / c_size), visualise=False))
        if not LIVE == 1:
            LABELS.append(label)
        else:
            LABELS.append('new0')
    # print('Elapsed Time on HoG = ' + str(time.time() - start_time) + '\n')


def test_from_disk(value):
    print ('Testing from disk')
    start_time = time.time()
    path = 'TST_IMGS/'
    global label
    global rotation
    global total
    global tst_dsk_percentage
    TOTAL = 0
    global failure
    FAILURE = 0
    for filename in os.listdir(path):
        TOTAL += 1
        if (TOTAL % 20) == 0:
            start_time = time.time()
        LABEL = filename.rsplit('_', 3)[0]
        ROTATION = int(filename.rsplit('_', 3)[1])
        # print ('Label ' + str(LABEL))
        # print 'Rotation ' + str(ROTATION)
        imagee = cv2.imread(path + filename)
        imagee = cv2.resize(imagee, (128,128))
        found_rot = get_img_rot(imagee)
        if not abs(ROTATION - found_rot) < 0.5:
            # print ('Testing ' + str(filename))
            FAILURE += 1
            # print ('Does not work')
            # cv2.imshow('Did not work',imagee)
            # cv2.waitKey(100)
            # print (found_rot)
            # print (ROTATION)
        if (TOTAL % 20) == 0:
            print('Elapsed Time Testing Image ' + str(TOTAL) + ' = ' + str(time.time() - start_time) + '\n')
    PERCENTAGE = 100 * FAILURE / TOTAL
    print ('Failure = ' + str(PERCENTAGE) + '%')
    print ('Failures = ' + str(FAILURE))
    print('Elapsed Time Testing = ' + str(time.time() - start_time) + '\n')
    print ('Done')


def live_learn(value):
    global live
    print (CLF)
    LIVE = 1


def learn_from_disk(value):
    path = 'LRN_IMGS/'
    global label
    i = 0
    for filename in os.listdir(path):
        if (i % 20) == 0:
            start_time = time.time()
        # print 'Learning ' + str(filename)
        LABEL = filename.rsplit('_', 2)[0]
        # print 'Label = ' + str(LABEL)
        imagee = cv2.imread(path + filename)
        learn_hog(imagee)
        if (i % 20) == 0:
            print('Elapsed Time Learning Image ' + str(i) + ' = ' + str(time.time() - start_time) + '\n')
        i += 1
    learn(1)
    print ('Done')


def big_test(value):
    global n_bin
    global b_size  # Align to cell size
    global c_size
    global b_stride  # Multiple of cell size 8 8
    global labels
    global hog_list
    global label
    global failure
    global total
    global tst_dsk_percentage
    for n_neighbors in range(11, 2, -1):
        global CLF
        CLF = KNeighborsClassifier(n_neighbors=n_neighbors)
        for bin_ in range(8, 2, -1):
            for block in (64, 128):
                for stride in (1, 2):
                    cell = block/2
                    B_STRIDE = cell*stride
                    start_time = time.time()
                    N_BIN = bin_
                    B_SIZE = block
                    C_SIZE = cell
                    LABELS = list()
                    HOG_LIST = list()
                    print ('Testing HoG')
                    print('neighbors = ' + str(n_neighbors) + '\n')
                    print('n_bin = ' + str(N_BIN) + '\n')
                    print('b_stride = ' + str(B_STRIDE) + '\n')
                    print('b_size = ' + str(B_SIZE) + '\n')
                    print('c_size = ' + str(C_SIZE) + '\n')
                    learn_from_disk(1)
                    test_from_disk(1)
                    with open('HoG_Trials.txt', 'a') as the_file:
                        the_file.write('neighbors = ' + str(n_neighbors) + '\n')
                        the_file.write('n_bin = ' + str(N_BIN) + '\n')
                        the_file.write('b_size = ' + str(B_SIZE) + '\n')
                        the_file.write('b_stride = ' + str(B_STRIDE) + '\n')
                        the_file.write('c_size = ' + str(C_SIZE) + '\n')
                        the_file.write('Failure = ' + str(FAILURE) + '\n')
                        the_file.write('Total = ' + str(TOTAL) + '\n')
                        the_file.write('Percentage = ' + str(PERCENTAGE) + '\n')
                        the_file.write('Elapsed Time = ' + str(time.time() - start_time) + '\n\n\n')
                    print('Written')
                    print ('Elapsed Time = ' + str(time.time() - start_time) + '\n')
    print ('Big Test Done')


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
    cv2.createTrackbar("Nb img profondeur", MAIN_WINDOW_NAME, nb_depth_imgs_cache, IMG_DEPTH_MAX, changeprofondeur)
    cv2.createTrackbar('Show', MAIN_WINDOW_NAME, 0, 1, show)
    cv2.createTrackbar("Show Color Image", MAIN_WINDOW_NAME, 0, 1, changeaffcouleur)
    cv2.createTrackbar("Show Depth Image", MAIN_WINDOW_NAME, 0, 1, changeaffprofondeur)
    cv2.createTrackbar('Learn from disk', MAIN_WINDOW_NAME, 0, 1, learn_from_disk)
    cv2.createTrackbar('Test from disk', MAIN_WINDOW_NAME, 0, 1, test_from_disk)
    cv2.createTrackbar('Save IMGs Learn', MAIN_WINDOW_NAME, 0, 1, save_imgs_learn)
    cv2.createTrackbar('Save IMGs Test', MAIN_WINDOW_NAME, 0, 1, save_imgs_test)
    cv2.createTrackbar('Info HoG', MAIN_WINDOW_NAME, 0, 1, hog_info)
    cv2.createTrackbar('Save HoG to Disk', MAIN_WINDOW_NAME, 0, 1, save_hog)
    cv2.createTrackbar('Learn', MAIN_WINDOW_NAME, 0, 1, learn)
    cv2.createTrackbar('Big Test', MAIN_WINDOW_NAME, 0, 1, big_test)
    cv2.createTrackbar('Load Class', MAIN_WINDOW_NAME, 0, 1, load_class)
    cv2.createTrackbar('Live Learn', MAIN_WINDOW_NAME, 0, 1, live_learn)
    cv2.createTrackbar('Debug', MAIN_WINDOW_NAME, 0, 1, debug)
    cv2.createTrackbar('Predict HoG', MAIN_WINDOW_NAME, 0, 1, hog_pred)
    cv2.createTrackbar("Depth Capture Range", MAIN_WINDOW_NAME, int(100 * val_depth_capture), 150, changecapture)
    cv2.imshow(MAIN_WINDOW_NAME, 0)
    print ("Creating subscribers")
    image_sub_rgb = rospy.Subscriber("/camera/rgb/image_rect_color", Image, callback_rgb, queue_size=1)
    image_sub_depth = rospy.Subscriber("/camera/depth_registered/image_raw/", Image, callback_depth, queue_size=1)
    detected_objects_list_publisher = rospy.Publisher('detected_objects_list', Detected_Objects_List, queue_size=1)
    print ("Spinning ROS")
    try:
        while not rospy.core.is_shutdown():
            rospy.rostime.wallsleep(0.5)
    except KeyboardInterrupt:
        print ("Shutting down")
        exit(1)
        cv2.destroyAllWindows()