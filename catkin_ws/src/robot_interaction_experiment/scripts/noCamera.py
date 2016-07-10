#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import svm
import tensorflow as tf

from collections import Counter
import os
import cv2
import pickle
import random
import time
import joblib
from skimage.feature import hog
import pylab as Plot
import tsne

####################
# Constants max HoG size = 900  #
####################

NB_MSG_MAX = 50
MAIN_WINDOW_NAME = "Segmentator"
MIN_AREA = 9000  # minimal area to consider a desirable object
MAX_AREA = 15000  # maximal area to consider a desirable object
N_COLORS = 80  # number of colors to consider when creating the image descriptor
IMG_DEPTH_MAX = 100  # maximum number of depth images to cache
LRN_PATH = 'LRN_IMGS/'
PARTIAL_LRN_PATH = 'PARTIAL_LRN_NOTEBOOK/'
TST_PATH = 'TST_IMGS/'
PARTIAL_TST_PATH = 'PARTIAL_TST_NOTEBOOK/'
TST_PATH_UPRIGHT = 'TST_UPRIGHT/'
NM_SAMPLES = 50000
####################
# Global variables #
####################

nb_depth_imgs_cache = 5  # default number of depth images to consider
show_depth = False
show_color = False
val_depth_capture = 0.03
depth_img_index = 0
last_depth_imgs = range(IMG_DEPTH_MAX + 1)
number_last_points = 15
hog_list = list()
depth_img_avg = 0
img_bgr8_clean = 0
got_color = False
got_depth = False
labels = list()
label = ''
loaded = 0
DEBUG = 0
show = 0
saving_learn = 0
img_clean_bgr_learn = None
img_clean_gray_class = None
saved = 0
color = ''
n_bin = 10  # 4 number of orientations for the HoG
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
hog_size = 0
# eta00 = 0.4
clf = SGDClassifier(loss='log', random_state=10)
# clf = svm.SVC(probability=True)
# clf = KNeighborsClassifier(n_neighbors=5)
implements_p_fit = 1
live_lrn_timer = 0
nb_real_additional_classes = 0
nb_reserved_classes = 20
loaded_clf = 0
speech = ''
got_speech = 0
img_buffer = list()
new_rgb = 0
timer_str = 0
random.seed(10)


def save_imgs_learn(value):
    global label
    global color
    global saving_learn
    global saved
    if isinstance(value, int):
        mode = str(input('Label: '))
        label = mode
        color_ = str(input('Color: '))
        color = color_
        saving_learn = 1
    else:
        cv2.imwrite('PARTIAL_LRN_NOTEBOOK/' + label + '_' + str(saved) + '_' + color + '.png', value)
        saved += 1
        print(saved)
        if saved == 60:
            saving_learn = 0
            saved = 0
            print('Done saving')


def save_imgs_test(value):
    global label
    global color
    global rotation
    global saving_test
    global saved
    if isinstance(value, int):
        mode = str(input('Label: '))
        label = mode
        color_ = str(input('Color: '))
        color = color_
        rotation = str(input('Rotation: '))
        saving_test = 1
    else:
        cv2.imwrite('PARTIAL_TST_NOTEBOOK/' + label + '_' + str(rotation) + '_' +
                    str(saved) + '_' + color + '.png', value)
        saved += 1
        print(saved)
        if saved == 20:
            saving_test = 0
            saved = 0
            print('Done saving')


def changecapture(n):
    global val_depth_capture
    if n == 0:
        n = 1
    val_depth_capture = float(n) / 100


def changeprofondeur(n):
    global nb_depth_imgs_cache
    nb_depth_imgs_cache = n
    if nb_depth_imgs_cache <= 0:
        nb_depth_imgs_cache = 1


def changeaffprofondeur(b):
    global show_depth
    if b == 1:
        show_depth = True
    else:
        show_depth = False


def changeaffcouleur(b):
    global show_color
    if b == 1:
        show_color = True
    else:
        show_color = False


def clean(img, n):
    # set the non-finite values (NaN, inf) to n
    # returns 1 where the img is finite and 0 where it is not
    mask = np.isfinite(img)
    #  where mask puts img, else puts n, so where is finite puts img, else puts n
    return np.where(mask, img, n)


'''
Callback from the audio related topic. Receives the current dictionary, words histogram and last spoken words (speech).
'''


def callback_audio_recognition(words):
    global speech
    global got_speech
    if not words.complete_words:
        return
    speech = words.complete_words
    got_speech = 1


def get_cube_upright():
    # Uses the depth image to only take the part of the image corresponding to the closest point and a bit further
    global depth_img_avg
    global img_bgr8_clean
    closest_pnt = np.amin(depth_img_avg)
    # resize the depth image so it matches the color one
    depth_img_avg = cv2.resize(depth_img_avg, (1280, 960))
    # generate a mask with the closest points
    img_detection = np.where(depth_img_avg < closest_pnt + val_depth_capture, depth_img_avg, 0)
    # put all the pixels greater than 0 to 255
    ret, mask = cv2.threshold(img_detection, 0.0, 255, cv2.THRESH_BINARY)
    # convert to 8-bit
    mask = np.array(mask, dtype=np.uint8)
    im2, contours, hierarchy = cv2.findContours(mask, 1, 2, offset=(0, -6))
    useful_cnts = list()
    uprightrects = list()
    img_bgr8_clean_copy = img_bgr8_clean.copy()
    for cnt in contours:
        if 9000 < cv2.contourArea(cnt) < 15000:
            if 420 < cv2.arcLength(cnt, 1) < 560:
                useful_cnts.append(cnt)
            else:
                print("Wrong Lenght 450 < " + str(cv2.arcLength(cnt, 1)) + str(" < 570"))
        else:
            print ("Wrong Area: 9000 < " + str(cv2.contourArea(cnt)) + " < 15000")
    for index, cnts in enumerate(useful_cnts):
        min_area_rect = cv2.minAreaRect(cnts)  # minimum area rectangle that encloses the contour cnt
        (center, size, angle) = cv2.minAreaRect(cnts)
        width, height = size[0], size[1]
        if not (0.7*height < width < 1.3*height):
            print("Wrong Height/Width: " + str(0.7*height) + " < " + str(width) + " < " + str(1.3*height))
            continue
        points = cv2.boxPoints(min_area_rect)  # Find four vertices of rectangle from above rect
        points = np.int32(np.around(points))  # Round the values and make it integers
        cv2.drawContours(img_bgr8_clean_copy, [points], 0, (0, 0, 255), 2)
        cv2.drawContours(img_bgr8_clean_copy, cnts, -1, (255, 0, 255), 2)
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
        uprightrects.append(uprightrect)
        uprightrect_copy = uprightrect.copy()
        cv2.drawContours(uprightrect_copy, [points], 0, (0, 0, 255), 2)
        cv2.imshow('uprightRect ' + str(index), uprightrect_copy)

    cv2.imshow('RBG', img_bgr8_clean_copy)
    cv2.waitKey(1)
    objects_detector(uprightrects)


def hog_pred(value):
    global n_bin
    global b_size
    global c_size
    global img_clean_gray_class
    opencv_hog = cv2.HOGDescriptor((128, 128), (b_size, b_size), (b_stride, b_stride), (c_size, c_size), n_bin,
                                   _L2HysThreshold=0.2)
    fd = opencv_hog.compute(img_clean_gray_class)
    fd = np.reshape(fd, (len(fd),))
    fd = fd.reshape(1, -1)
    print(fd)
    print(np.shape(fd))
    global clf
    print(clf.predict(fd))
    print(clf.predict_proba(fd))


# noinspection PyBroadException
def load_hog(value):
    global clf
    global hog_list
    global labels
    global shuffled_x
    global shuffled_y
    # try:
    f = open('HOG_N_LABELS/HOG_N_LABELS.pickle', 'rb')
    hog_tuple = pickle.load(f)
    # except:
    #     print("Problem loading HoG, is the file there?")
    #     return -1
    hog_list = hog_tuple[0]
    labels = hog_tuple[1]
    global loaded
    loaded = 1
    print('Loaded')
    database_indexs = list(range(len(labels)))
    random.shuffle(database_indexs)
    shuffled_x = [hog_list[i] for i in database_indexs]
    shuffled_y = [labels[i] for i in database_indexs]
    return 1


def plot_2d_classes(value):
    global labels
    if 10 > len(labels):
        print ("Labels and HoG dont seem to have been loaded")
        print ("Trying to load them from disk")
        if not load_hog(1) == 1:
            print ("Could not load HoG, quitting")
            return
    nm_elements = int(input('Plot this many elements (up to ' + str(len(labels)) + ') : '))
    new_labels = list()
    classes = list()
    classes = np.unique(labels).tolist()
    for labell in labels[:nm_elements]:
        for unique_label in classes:
            if unique_label == labell:
                new_labels.append(classes.index(unique_label))
    y = tsne.tsne(np.array(hog_list[:nm_elements]))
    Plot.scatter(y[:, 0], y[:, 1], 20, new_labels)
    Plot.show()


def show_imgs(value):
    global show
    show = value


def debug(value):
    global DEBUG
    DEBUG = value


def learn(value):
    global hog_list
    global labels
    global shuffled_x
    global shuffled_y
    global loaded_clf
    global scaler
    print('Learning')
    lrn_start_time = time.time()
    classes = np.unique(labels).tolist()
    if implements_p_fit == 1:
        for i in list(range(nb_reserved_classes)):
            classes.append('new' + str(i))
    print(classes)
    database_indexs = list(range(len(labels)))
    if implements_p_fit == 1:
        for i in list(range(7)):
            print("Pass " + str(i) + " of " + "6")
            random.shuffle(database_indexs)
            # scaler = StandardScaler()
            shuffled_x = [hog_list[i] for i in database_indexs]
            shuffled_y = [labels[i] for i in database_indexs]
            # scaler.fit(shuffled_x)  # Don't cheat - fit only on training data
            # shuffled_x = scaler.transform(shuffled_x)
            nb_range = 1
            for i2 in list(range(nb_range)):
                clf.partial_fit(shuffled_x[int(i2 * len(shuffled_x) / nb_range):int((i2 + 1) * len(shuffled_x) / nb_range)],
                                shuffled_y[int(i2 * len(shuffled_x) / nb_range):int((i2 + 1) * len(shuffled_x) / nb_range)]
                                , classes)
    else:
        database_indexs = list(range(len(labels)))
        random.shuffle(database_indexs)
        shuffled_x = [hog_list[i] for i in database_indexs]
        shuffled_y = [labels[i] for i in database_indexs]
        clf.fit(shuffled_x, shuffled_y)
    loaded_clf = 1
    print('Done Learning')
    print('Elapsed Time Learning = ' + str(time.time() - lrn_start_time) + '\n')


def save_hog(value):
    global clf
    hog_tuple = (hog_list, labels)
    print('labels = ' + str(np.unique(hog_tuple[1])))
    with open('HOG_N_LABELS/HOG_N_LABELS.pickle', 'wb') as f:
        pickle.dump(hog_tuple, f)
    print('Done')


def save_classifier(value):
    global clf
    joblib.dump(clf, 'Classifier/filename.pkl')
    print('Done')


# noinspection PyBroadException
def load_classifier(value):
    global clf
    global loaded_clf
    try:
        clf = joblib.load('Classifier/filename.pkl')
    except:
        print ("Loading failed")
        loaded_clf = 0
        return -1
    loaded_clf = 1
    print ("Loaded Classifier")
    return 1


def hog_info(value):
    global labels
    global hog_list
    print('Current labels = ')
    myset = set(labels)
    print(str(myset))
    print('Current HoG size:')
    print(len(hog_list))
    print("Single HoG size: ")
    print(len(hog_list[0]))
    print (Counter(labels))




def live_learn(value):
    global live
    global live_cnt
    global live_lrn_timer
    global nb_depth_imgs_cache
    global hog_list
    global labels
    global nb_real_additional_classes
    global shuffled_x
    global shuffled_y
    global hog_size
    global clf
    if isinstance(value, int):
        hog_size = len(labels)
        if hog_size < 10:
            print ("HoGs not yet loaded")
            print ("Trying to load it")
            load_hog(1)
        if loaded_clf == 0:
            print("Classifier not fitted, trying to load one")
            if load_classifier(1) == -1:
                print("Could not load it, quitting")
        hog_size = len(labels)
        live_lrn_timer = time.time()
        live = 1
        nb_real_additional_classes += 1
        print ("Current HoG size = " + str(hog_size))
        database_indexs = list(range(len(labels)))
        random.shuffle(database_indexs)
        shuffled_x = [hog_list[i] for i in database_indexs]
        shuffled_y = [labels[i] for i in database_indexs]
    else:
        img_bgr8 = value
        if live_cnt == 0:
            live_cnt = 100
            nb_depth_imgs_cache = 1
            hog_list = list()
            labels = list()
        learn_hog(img_bgr8)
        shuffledrange = list(range(hog_size))
        print(live_cnt)
        live_cnt -= 1
        if live_cnt == 0:
            hog_list.extend(shuffled_x[1:(int(hog_size * float(1.0/(len(np.unique(shuffled_y))-nb_reserved_classes + nb_real_additional_classes))))])
            labels.extend(shuffled_y[1:(int(hog_size * float(1.0/(len(np.unique(shuffled_y))-nb_reserved_classes + nb_real_additional_classes))))])
            shuffledrange = list(range(len(hog_list)))
            print ((int(hog_size * float(1.0/(len(np.unique(shuffled_y))-nb_reserved_classes+nb_real_additional_classes)))))
            print (len(hog_list))
            for passes in list(range(5)):
                shuffledx_temp = [hog_list[i] for i in shuffledrange]
                shuffledy_temp = [labels[i] for i in shuffledrange]
                start_time = time.time()
                clf.partial_fit(shuffledx_temp, shuffledy_temp)
                print('Elapsed Time LEARNING = ' + str(time.time() - start_time) + '\n')
            live = 0
            nb_depth_imgs_cache = 5
            print('Elapsed Time TOTAL = ' + str(time.time() - live_lrn_timer) + ' FPS = ' + str(100 / (time.time() - live_lrn_timer)) + '\n')


def get_img_to_be_sent(img):
    global hog_size
    hog_size = len(labels)
    if hog_size < 10:
        print("HoGs not yet loaded")
        print("Trying to load it")
        load_hog(1)
    if loaded_clf == 0:
        print("Classifier not fitted, trying to load one")
        if load_classifier(1) == -1:
            print("Could not load it, quitting")
    img_rotation, confiance = get_img_rot(img)
    if not img_rotation == 0:
        rows, cols, d = img.shape
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), img_rotation * 90, 1)
        img = cv2.warpAffine(img, m, (cols, rows))
    return cv2.resize(img, (256, 256)), confiance


def objects_detector(imgs_bgr8):
    global img_clean_gray_class
    global img_clean_bgr_learn
    global clf
    global n_bin
    global b_size
    global c_size
    global saving_learn
    global saved
    global saving_test
    global live
    global show
    global got_speech
    global speech
    global loaded_clf
    detected_objects_list = []
    objects_detector_time = time.time()
    for index, img_bgr8 in enumerate(imgs_bgr8):
        width, height, d = np.shape(img_bgr8)
        w, l, d = np.shape(img_bgr8)
        img_clean_bgr_learn = img_bgr8.copy()
        img_bgr8 = img_bgr8[13:w - 5, 13:l - 8]
        img_clean_bgr_class = img_bgr8.copy()
        img_clean_bgr_class = cv2.resize(img_clean_bgr_class, (128, 128), interpolation=cv2.INTER_AREA)  # resize image
        img_clr = cv2.resize(img_clean_bgr_learn, (128, 128))
        img_gray = cv2.resize(cv2.cvtColor(img_clean_bgr_learn, cv2.COLOR_BGR2GRAY), (128, 128))
        cv2.imshow('img_gray', img_gray)
        cv2.imshow('Clean' + str(index), img_clean_bgr_class)
        if loaded_clf:
            final, confiance = get_img_to_be_sent(img_clean_bgr_class)
            print("storing " + str(confiance))

            if confiance < 0.85:
                str_img(img_clean_bgr_learn)
        if saving_learn == 1:
            save_imgs_learn(img_clean_bgr_learn)
        if saving_test == 1:
            save_imgs_test(img_clean_bgr_class)
        if live == 1:
            live_learn(img_bgr8)
        if show:
            if loaded_clf == 0:
                print("Classifier not fitted, trying to load one")
                if load_classifier(1) == -1:
                    print("Could not load it, quitting")
            else:
                cv2.imshow('Sent' + str(index), final)
        rows, cols, d = img_clean_bgr_class.shape
    if got_speech == 0:
        return
    got_speech = 0


def get_img_rot(img_bgr):
    global scaler
    best_rot = 0
    best_perc = 0
    for i in list(range(4)):
        fd = get_img_hog(img_bgr)
        # fd = scaler.transform(fd)  # apply same transformation to test data
        for percentage in clf.predict_proba(fd)[0]:
            if percentage > best_perc:
                best_perc = percentage
                best_rot = i
        rows, cols, d = img_bgr.shape
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img_bgr = cv2.warpAffine(img_bgr, m, (cols, rows))
    return best_rot, best_perc


def get_img_hog(img_bgr):
    # img_gry = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    opencv_hog = cv2.HOGDescriptor((128, 128), (b_size, b_size), (b_stride, b_stride), (c_size, c_size), n_bin, _L2HysThreshold=0.2)
    h1 = opencv_hog.compute(img_bgr)
    fd = np.reshape(h1, (len(h1),))
    fd = fd.reshape(1, -1)
    return fd


def learn_hog(img):
    start_time = time.time()
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    global n_bin
    global b_size
    global c_size
    global hog_list
    global labels
    global label
    global show
    global nb_real_additional_classes
    w, l, d = np.shape(img)
    img_list = list()
    img_list.append((img[:, :]))  # no changes
    global live
    if live == 0:
        for i in list(range(1, 20, 2)):
            img_list.append((img[0:w - i, :]))  # cut right
            img_list.append((img[i:, :]))  # cut left
            img_list.append((img[:, i:]))  # cut up
            img_list.append((img[:, 0:l - i]))  # cut down
            img_list.append((img[:, i:l - i]))  # cut up and down
            img_list.append((img[i:w - i, :]))  # cut left and right
            img_list.append((img[i:, i:l - i]))  # cut up and down and left
            img_list.append((img[:w - i, i:l - i]))  # cut up and down and right
            img_list.append((img[i:w - i, i:l - i]))  # cut up and down and left and right
    else:
        for i in list(range(1, 20, 2)):
            img_list.append((img[0:w - i, :]))  # cut right
            img_list.append((img[i:, :]))  # cut left
            img_list.append((img[:, i:]))  # cut up
            img_list.append((img[:, 0:l - i]))  # cut down
            img_list.append((img[:, i:l - i]))  # cut up and down
            img_list.append((img[i:w - i, :]))  # cut left and right
            img_list.append((img[i:, i:l - i]))  # cut up and down and left
            img_list.append((img[:w - i, i:l - i]))  # cut up and down and right
            img_list.append((img[i:w - i, i:l - i]))  # cut up and down and left and right
    index = 0
    start_time = time.time()
    opencv_hog = cv2.HOGDescriptor((128, 128), (b_size, b_size), (b_stride, b_stride), (c_size, c_size), n_bin, _L2HysThreshold=0.2)
    for index,imgs in enumerate(img_list):
        imgs = cv2.resize(imgs, (128, 128), interpolation=cv2.INTER_AREA)  # resize image
        # cv2.imshow('LRN' + str(index),imgs)
        # cv2.waitKey(500)
        if show == 1:
            cv2.imshow('img' + str(index), imgs)
        index += 1
        h1 = opencv_hog.compute(imgs)
        h1 = (np.reshape(h1, (len(h1),)))
        hog_list.append((np.reshape(h1, (len(h1),))))
        # print (hog_list)
        # hog_list.append(hog(imgs, orientations=n_bin, pixels_per_cell=(c_size, c_size),
        #                     cells_per_block=(b_size / c_size, b_size / c_size), visualise=False))
        if not live == 1:
            labels.append(label)
        else:
            labels.append('new' + str(nb_real_additional_classes))
    # print ('new' + str(nb_real_additional_classes))


def partial_lrn_disk(value):
    global label
    global PARTIAL_LRN_PATH
    global loaded_clf
    global live
    global live_cnt
    global live_lrn_timer
    global nb_depth_imgs_cache
    global hog_list
    global labels
    global nb_real_additional_classes
    global shuffled_x
    global shuffled_y
    global hog_size
    global clf
    live = 1
    i = 0
    hog_list = list()
    labels = list()
    for filename in os.listdir(PARTIAL_LRN_PATH):
        if filename.endswith(".png"):
            if (i % 20) == 0:
                start_time = time.time()
            label = filename.rsplit('_', 2)[0]
            image_read = cv2.imread(PARTIAL_LRN_PATH + filename)
            learn_hog(image_read)
            if (i % 40) == 0:
                print('Elapsed Time Learning Image ' + str(i) + ' = ' + str(time.time() - start_time))
            i += 1
        # shuffled_x.extend(hog_list)
        # shuffled_y.extend(labels)
    hog_list.extend(
            shuffled_x[1:int(0.4*len(shuffled_y))])
    labels.extend(
            shuffled_y[1:int(0.4*len(shuffled_y))])
    for passes in list(range(5)):
        shuffledrange = list(range(len(hog_list)))
        random.shuffle(shuffledrange)
        shuffledx_temp = [hog_list[i] for i in shuffledrange]
        shuffledy_temp = [labels[i] for i in shuffledrange]
        start_time = time.time()
        clf.partial_fit(shuffledx_temp, shuffledy_temp)
        print('Elapsed Time LEARNING = ' + str(time.time() - start_time) + '\n')
    live = 0
    loaded_clf = 1
    print('Done')
    return


def partial_test_from_disk(value):
    print('Testing from disk')
    start_time = time.time()
    lowest_conf = 10000
    max_conf = 0
    global PARTIAL_TST_PATH
    global label
    global rotation
    global total
    global tst_dsk_percentage
    total = 0
    global failure
    failure = 0
    for filename in os.listdir(PARTIAL_TST_PATH):
        total += 1
        if (total % 100) == 0:
            start_time = time.time()
        label = filename.rsplit('_', 3)[0]
        rotation = int(filename.rsplit('_', 3)[1])
        image = cv2.imread(PARTIAL_TST_PATH + filename)
        image = cv2.resize(image, (128, 128))
        found_rot, confiance = get_img_rot(image)
        if confiance < lowest_conf:
            lowest_conf = confiance
        if confiance > max_conf:
            max_conf = confiance
        # if confiance < 0.8:
        #     print(confiance)
        #     cv2.imshow('unsure', image)
        #     cv2.waitKey(1000)
        # if confiance < 0.9:
        #     print (confiance)
        #     cv2.imshow('unsure', image)
        #     cv2.waitKey(1000)
        if not abs(rotation - found_rot) < 0.5:
            failure += 1
            # cv2.imshow('MISTAKE', image)
            # cv2.waitKey(1000)
        if (total % 100) == 0:
            print('Elapsed Time Testing Image ' + str(total) + ' = ' + str(time.time() - start_time))
    tst_dsk_percentage = 100 * failure / total
    print('Failure = ' + str(tst_dsk_percentage) + '%')
    print('Failures = ' + str(failure))
    print('Elapsed Time Testing = ' + str(time.time() - start_time) + '\n')
    print('Done')
    print("Lowest conf = " + str(lowest_conf))
    print("Max conf = " + str(max_conf))



def test_from_disk(value):
    print('Testing from disk')
    start_time = time.time()
    global TST_PATH
    global label
    global rotation
    global total
    global tst_dsk_percentage
    total = 0
    global failure
    failure = 0
    lowest_conf = 10000
    for filename in os.listdir(TST_PATH):
        total += 1
        if (total % 100) == 0:
            start_time = time.time()
        label = filename.rsplit('_', 3)[0]
        rotation = int(filename.rsplit('_', 3)[1])
        image = cv2.imread(TST_PATH + filename)
        image = cv2.resize(image, (128, 128))
        found_rot, confiance = get_img_rot(image)
        if confiance < lowest_conf:
            lowest_conf = confiance
        # if confiance < 0.8:
        #     print (confiance)
        #     cv2.imshow('unsure', image)
        #     cv2.waitKey(1000)
        if not abs(rotation - found_rot) < 0.5:
            failure += 1
            # print (confiance)
            # img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow('MISTAKE', image)
            # print(clf.predict(get_img_hog(image)))
            cv2.waitKey(1)
        if (total % 800) == 0:
            print('Elapsed Time Testing Image ' + str(total) + ' = ' + str(time.time() - start_time))
    tst_dsk_percentage = 100 * failure / total
    print('Failure = ' + str(tst_dsk_percentage) + '%')
    print('Failures = ' + str(failure))
    print('Elapsed Time Testing = ' + str(time.time() - start_time) + '\n')
    print('Done')
    print("Lowest conf = " + str(lowest_conf))


def test_from_disk_label(value):
    print('Testing from disk')
    start_time = time.time()
    global TST_PATH_UPRIGHT
    global label
    global rotation
    global total
    global tst_dsk_percentage
    total = 0
    global failure
    failure = 0
    for filename in os.listdir(TST_PATH_UPRIGHT):
        total += 1
        if (total % 20) == 0:
            start_time = time.time()
        label = filename.rsplit('_', 3)[0]
        imagee = cv2.imread(TST_PATH_UPRIGHT + filename)
        imagee = cv2.resize(imagee, (128, 128))
        opencv_hog = cv2.HOGDescriptor((128, 128), (b_size, b_size), (b_stride, b_stride), (c_size, c_size), n_bin,
                                       _L2HysThreshold=0.2)
        h1 = opencv_hog.compute(imagee)
        fd = np.reshape(h1, (len(h1),))
        fd = fd.reshape(1, -1)
        found_label = clf.predict(fd)[0]
        if not str(found_label) == str(label):
            failure += 1
            print("Got " + str(found_label))
            print("Expected " + str(label))
            print("\n")
    tst_dsk_percentage = 100 * failure / total
    print('Failure = ' + str(tst_dsk_percentage) + '%')
    print('Failures = ' + str(failure))
    print('Elapsed Time Testing = ' + str(time.time() - start_time) + '\n')
    print('Done')


def learn_from_disk(value):
    global label
    global LRN_PATH
    global loaded_clf
    i = 0
    for filename in os.listdir(LRN_PATH):
        if filename.endswith(".png"):
            if (i % 500) == 0:
                start_time = time.time()
            label = filename.rsplit('_', 2)[0]
            image_read = cv2.imread(LRN_PATH + filename)
            learn_hog(image_read)
            if (i % 500) == 0:
                print('Elapsed Time Learning Image ' + str(i) + ' = ' + str(time.time() - start_time))
            i += 1
    learn(1)
    loaded_clf = 1
    print('Done')


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
    global clf
    for bin_ in list(range(16, 2, -1)):
        for block in (32, 64, 128):
            for stride in (1, 2, 4, 8):
                cell = block / 2
                b_stride = cell / stride
                start_time = time.time()
                n_bin = bin_
                b_size = block
                c_size = cell
                labels = list()
                hog_list = list()
                print('Testing HoG')
                print('n_bin = ' + str(n_bin) + '\n')
                print('b_stride = ' + str(b_stride) + '\n')
                print('b_size = ' + str(b_size) + '\n')
                print('c_size = ' + str(c_size) + '\n')
                learn_from_disk(1)
                test_from_disk(1)
                with open('HoG_Trials.txt', 'a') as the_file:
                    the_file.write('n_bin = ' + str(n_bin) + '\n')
                    the_file.write('b_size = ' + str(b_size) + '\n')
                    the_file.write('b_stride = ' + str(b_stride) + '\n')
                    the_file.write('c_size = ' + str(c_size) + '\n')
                    the_file.write('single HoG size = ' + str(len(hog_list[0])) + '\n')
                    the_file.write('Failure = ' + str(failure) + '\n')
                    the_file.write('Total = ' + str(total) + '\n')
                    the_file.write('Percentage = ' + str(tst_dsk_percentage) + '\n')
                    the_file.write('Elapsed Time = ' + str(time.time() - start_time) + '\n\n\n')
                print('Written')
                print('Elapsed Time = ' + str(time.time() - start_time) + '\n')
                clf = SGDClassifier(loss='log')
    print('Big Test Done')


if __name__ == '__main__':
    print("Creating windows")
    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Nb img profondeur", MAIN_WINDOW_NAME, nb_depth_imgs_cache, IMG_DEPTH_MAX, changeprofondeur)
    cv2.createTrackbar('Show', MAIN_WINDOW_NAME, 0, 1, show_imgs)
    cv2.createTrackbar("Show Color Image", MAIN_WINDOW_NAME, 0, 1, changeaffcouleur)
    cv2.createTrackbar("Show Depth Image", MAIN_WINDOW_NAME, 0, 1, changeaffprofondeur)
    cv2.createTrackbar('Learn from disk', MAIN_WINDOW_NAME, 0, 1, learn_from_disk)
    cv2.createTrackbar('Test from disk', MAIN_WINDOW_NAME, 0, 1, test_from_disk)
    cv2.createTrackbar('Partial Fit', MAIN_WINDOW_NAME, 0, 1, partial_lrn_disk)
    cv2.createTrackbar('Partial TST', MAIN_WINDOW_NAME, 0, 1, partial_test_from_disk)
    cv2.createTrackbar('Save IMGs Learn', MAIN_WINDOW_NAME, 0, 1, save_imgs_learn)
    cv2.createTrackbar('Save IMGs Test', MAIN_WINDOW_NAME, 0, 1, save_imgs_test)
    cv2.createTrackbar('Info HoG', MAIN_WINDOW_NAME, 0, 1, hog_info)
    cv2.createTrackbar('Save HoG to Disk', MAIN_WINDOW_NAME, 0, 1, save_hog)
    cv2.createTrackbar('Learn HoG stored in Memory', MAIN_WINDOW_NAME, 0, 1, learn)
    cv2.createTrackbar('Test labels from disk', MAIN_WINDOW_NAME, 0, 1, test_from_disk_label)
    cv2.createTrackbar('Load HoG', MAIN_WINDOW_NAME, 0, 1, load_hog)
    cv2.createTrackbar('Save Classifier', MAIN_WINDOW_NAME, 0, 1, save_classifier)
    cv2.createTrackbar('Load Classifier', MAIN_WINDOW_NAME, 0, 1, load_classifier)
    cv2.createTrackbar('Plot Classes as 2D', MAIN_WINDOW_NAME, 0, 1, plot_2d_classes)
    cv2.createTrackbar('Debug', MAIN_WINDOW_NAME, 0, 1, debug)
    cv2.createTrackbar('Predict HoG', MAIN_WINDOW_NAME, 0, 1, hog_pred)
    cv2.createTrackbar("Depth Capture Range", MAIN_WINDOW_NAME, int(100 * val_depth_capture), 150, changecapture)
    # load_classifier(1)
    # load_hog(1)
    learn_from_disk(1)
    test_from_disk(1)
    # partial_lrn_disk(1)
    # test_from_disk(1)
    # partial_test_from_disk(1)
    cv2.imshow(MAIN_WINDOW_NAME, 0)
    cv2.waitKey(0)
    # big_test(1)
    print("Creating subscribers")
    print("Spinning ROS")
