#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
import random
from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
import pickle
import os

import bufferImageMsg

#############
# Constants #
#############

WIDTH = 1280
HEIGHT = 960

NB_MSG_MAX = 50
NB_MAX_DEPTH_IMG = 30

NB_IMG_FOR_LEARN = 30

LEARN_PATH = '../../LRN_IMGS/'
TEST_PATH = '../../TST_IMGS/'

MAIN_WINDOW = "Controls"

SHOW_COLOR_IMG = "Show color img"
SHOW_DEPTH_IMG = "Show depth img"

LEARN_FROM_DISK = "Learn"
TEST_FROM_DISK = "Test"
UNKNOWN_OBJECT = "Unknown?"

RUN_NAME = "Run"


#########
# Class #
#########

class ImgAveraging:
    def __init__(self, nb_max):
        self.__NB_MAX_REF = nb_max
        self.__NB_MAX = nb_max
        self.__nb_stock = 0
        self.__first = 0
        self.__last = 0
        self.__container = [0 for _ in range(nb_max)]
        self.__img_sum = 0

    def average(self):
        av = np.array([[0]])

        if self.__nb_stock > 0:
            av = self.__img_sum / self.__nb_stock

        return av

    def new_capacity(self, nb):
        # redefined object
        if self.__NB_MAX_REF >= nb > 0:
            self.__NB_MAX = nb
            self.__nb_stock = 0
            self.__first = 0
            self.__last = 0
            self.__img_sum = 0

    def new_img(self, img):
        if self.__nb_stock == self.__NB_MAX:
            self.__img_sum -= self.__peek()

        self.__img_sum += img
        self.__push(img)

    def __peek(self):
        item = None

        if self.__nb_stock > 0:
            item = self.__container[self.__first]

        return item

    def __pop(self):
        item = None

        if self.__nb_stock > 0:
            item = self.__container[self.__first]
            self.__first += 1
            self.__first %= self.__NB_MAX
            self.__nb_stock -= 1

        return item

    def __push(self, img):
        # erase first item if the container is full
        if self.__nb_stock == self.__NB_MAX:
            self.__pop()

        self.__container[self.__last] = img
        self.__last += 1
        self.__last %= self.__NB_MAX
        self.__nb_stock += 1


####################
# Global variables #
####################

b_size = 64  # 15  # block size
c_size = 32  # 15  # cell size
callback_rgb_timer = 0
clf = SGDClassifier(loss='log')
color = ''
color_buffer_msg = bufferImageMsg.BufferImageMsg(NB_MSG_MAX)
color_img = np.array([[0]])
debug_flag = 1
depth_buffer_msg = bufferImageMsg.BufferImageMsg(1)
depth_capture = 0.03
depth_img = np.array([[0]])
depth_img_averaging = ImgAveraging(NB_MAX_DEPTH_IMG)
hog_list = list()
img_bgr8_clean = np.array([[0]])
img_clean_bgr_learn = np.array([[0]])
img_clean_gray_class = np.array([[0]])
interactions_flag = 0
label = ''
labels = list()
last_hog = 0
learn_flag = 0
live_cnt = 0
live_flag = 0
loaded_flag = 0
n_bin = 6  # 4  # number of orientations for the HoG
nb_depth_img = 30
recording_flag = 0
rotation = 0
run_flag = 0
saved_flag = 0
saving_learn = 0
saving_test = 0
show_color_img_flag = 0
show_depth_img_flag = 0
show_flag = 0
shuffled_x = list()
shuffled_y = list()
start_time2 = 0

# __TEST__
begin_learn = 0
end_learn = 0


######################
# Callback functions #
######################

def depth_capture_callback(n):
    global depth_capture
    if n == 0:
        n = 1
    depth_capture = float(n) / 100


def learn_from_disk_callback(value):
    global begin_learn
    global learn_flag
    global show_flag

    if value == 1:
        learn_from_disk()
        # show_flag = 1

    learn_flag = value


def run_callback(value):
    global run_flag
    run_flag = value


def main_callback(_):
    get_color_img()
    get_depth_img()

    show_color_img()
    show_depth_img()

    if run_flag == 1:
        filter_by_depth()
    cv2.waitKey(1)


def nb_depth_img_callback(n):
    global nb_depth_img
    nb_depth_img = n
    if nb_depth_img <= 0:
        nb_depth_img = 1


def show_color_img_callback(value):
    global show_color_img_flag
    show_color_img_flag = value


def show_depth_img_callback(value):
    global show_depth_img_flag
    show_depth_img_flag = value


def test_from_disk_callback(value):
    if value == 1:
        test_from_disk()


def unknown_object_callback(value):
    if value == 1:
        color_buffer_msg.run(NB_IMG_FOR_LEARN)


#############
# Functions #
#############

def clean(img, n):
    # set the non-finite values (NaN, inf) to n
    # returns 1 where the img is finite and 0 where it is not
    mask = np.isfinite(img)
    #  where mask puts img, else puts n, so where is finite puts img, else puts n
    return np.where(mask, img, n)


def filter_by_depth():
    # Uses the depth image to only take the part of the image corresponding to the closest point and a bit further
    global depth_img_averaging

    depth_img_avg = depth_img_averaging.average()
    closest_pnt = np.amin(depth_img_avg)
    # print np.shape(depth_img_avg)
    depth_img_avg = cv2.resize(depth_img_avg, (WIDTH, HEIGHT))
    depth_range = closest_pnt + depth_capture
    # print np.shape(depth_img_avg)
    # generate a mask with the closest points
    img_detection = np.where(depth_img_avg < depth_range, depth_img_avg, 0)
    # put all the pixels greater than 0 to 255
    ret, mask = cv2.threshold(img_detection, 0.0, 255, cv2.THRESH_BINARY)
    mask = np.array(mask, dtype=np.uint8)  # convert to 8-bit
    im2, contours, hierarchy = cv2.findContours(mask, 1, 2, offset=(0, -6))
    biggest_cont = contours[0]
    # biggest_cont = np.array([[0, 1], [1, 1], [0, 0]]) # smallest contour
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
    cv2.namedWindow('RBG', cv2.WINDOW_NORMAL)
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
    img_width, img_height = (img_bgr8_clean.shape[0], img_bgr8_clean.shape[1])
    rotated = cv2.warpAffine(img_bgr8_clean, rot_matrix, (img_height, img_width), flags=cv2.INTER_CUBIC)
    # extract the rect after rotation has been done
    size_int = (np.int32(size[0]), np.int32(size[1]))
    up_right_rect = cv2.getRectSubPix(rotated, size_int, center)
    # up_right_rect_copy = up_right_rect.copy()
    # cv2.drawContours(up_right_rect_copy, [points], 0, (0, 0, 255), 2)
    # cv2.imshow('uprightRect', up_right_rect_copy)
    objects_detector(up_right_rect)


def get_color_img():
    global color_buffer_msg, color_img, img_bgr8_clean

    msg = color_buffer_msg.get_last_img_msg()
    if msg is not None:
        color_img = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        img_bgr8_clean = color_img[32:992, 0:1280]


def get_depth_img():
    global depth_buffer_msg, depth_img, depth_img_averaging

    msg = depth_buffer_msg.get_last_img_msg()
    if msg is not None:
        clean_img = clean(CvBridge().imgmsg_to_cv2(msg, "passthrough"), 255)
        depth_img = clean_img
        depth_img_averaging.new_img(depth_img)


def get_img_rot(img_bgr):
    img_clean_gray_class_local = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    best_rot = 0
    best_perc = 0
    # noinspection PyArgumentList
    opencv_hog = cv2.HOGDescriptor((128, 128), (b_size, b_size), (c_size, c_size), (c_size, c_size), n_bin)
    for i in range(4):
        # Calculate HoG
        h1 = opencv_hog.compute(img_clean_gray_class_local)
        fd = (np.reshape(h1, (len(h1),)))
        fd = fd.reshape(1, -1)
        for percentage in clf.predict_proba(fd)[0]:
            if percentage > best_perc:
                best_perc = percentage
                best_rot = i
        rows, cols = img_clean_gray_class_local.shape
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img_clean_gray_class_local = cv2.warpAffine(img_clean_gray_class_local, m, (cols, rows))
    return best_rot


def hog_appender():
    global recording_flag
    global last_hog
    global hog_list
    global labels
    global label

    print ('Already have these labels:')
    myset = set(labels)
    print (str(myset))
    mode = str(raw_input('Label: '))
    label = mode
    recording_flag = 1


def hog_info():
    global labels
    global hog_list
    print ('Current labels = ')
    myset = set(labels)
    print (str(myset))
    print ('Current HoG size:')
    print (len(hog_list))


def hog_pred():
    global n_bin
    global b_size
    global c_size
    global img_clean_gray_class
    global clf

    fd = hog(img_clean_gray_class, orientations=n_bin, pixels_per_cell=(c_size, c_size),
             cells_per_block=(b_size / c_size, b_size / c_size), visualise=False)
    print (clf.predict([fd]))


def learn():
    global hog_list
    global shuffled_x
    global shuffled_y

    print ('Learning')
    start_time = rospy.get_time()
    classes = np.unique(labels).tolist()
    for i in range(10):
        classes.append('new' + str(i))
    print (classes)

    shuffledrange = range(len(labels))
    for i in range(5):
        random.shuffle(shuffledrange)
        shuffled_x = [hog_list[i] for i in shuffledrange]
        shuffled_y = [labels[i] for i in shuffledrange]
        print (len(shuffled_x))
        for i2 in range(10):
            print (i2)
            clf.partial_fit(shuffled_x[i2 * len(shuffled_x) / 10:(i2 + 1) * len(shuffled_x) / 10],
                            shuffled_y[i2 * len(shuffled_x) / 10:(i2 + 1) * len(shuffled_x) / 10], classes)
    print ('Done Learning')
    print('Elapsed Time Learning = ' + str(rospy.get_time() - start_time) + '\n')


def learn_from_disk():
    global label
    i = 0
    start_time = 0
    for filename in os.listdir(LEARN_PATH):
        if (i % 20) == 0:
            start_time = rospy.get_time()
        imagee = cv2.imread(LEARN_PATH + filename)
        learn_hog(imagee)
        if (i % 20) == 0:
            print('Elapsed Time Learning Image ' + str(i) + ' = ' + str(rospy.get_time() - start_time) + '\n')
        i += 1
    learn()
    print ('Done')


def learn_hog(img):
    global n_bin
    global b_size
    global c_size
    global hog_list
    global labels
    global live_flag
    global show_flag

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, l = np.shape(img)
    img_list = list()
    img_list.append((img[:, :]))  # no changes

    if live_flag == 0:
        for i in range(1, 12, 2):
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
        for i in range(3, 12, 3):
            img_list.append((img[0:w - i, :]))  # cut right
            img_list.append((img[i:, :]))  # cut left
            img_list.append((img[:, i:]))  # cut up
            img_list.append((img[:, 0:l - i]))  # cut down
            img_list.append((img[:, i:l - i]))  # cut up and down
            img_list.append((img[i:w - i, :]))  # cut left and right
            img_list.append((img[i:, i:l - i]))  # cut up and down and left
            img_list.append((img[:w - i, i:l - i]))  # cut up and down and right
            img_list.append((img[i:w - i, i:l - i]))  # cut up and down and left and right
    # hog = cv2.HOGDescriptor()
    index = 0
    # print('Elapsed Time Pre HoG = ' + str(time.time() - start_time) + '\n')
    # noinspection PyArgumentList
    opencv_hog = cv2.HOGDescriptor((128, 128), (b_size, b_size), (c_size, c_size), (c_size, c_size), n_bin)
    for imgs in img_list:
        imgs = cv2.resize(imgs, (128, 128), interpolation=cv2.INTER_AREA)  # resize image
        if show_flag == 1:
            cv2.imshow('img' + str(index), imgs)
        index += 1
        h1 = opencv_hog.compute(imgs)
        h1 = (np.reshape(h1, (len(h1),)))
        hog_list.append((np.reshape(h1, (len(h1),))))
        # print (HOG_LIST)
        # HOG_LIST.append(hog(imgs, orientations=n_bin, pixels_per_cell=(c_size, c_size),
        #                     cells_per_block=(b_size / c_size, b_size / c_size), visualise=False))
        if not live_flag == 1:
            labels.append(label)
        else:
            labels.append('new0')
            # print('Elapsed Time on HoG = ' + str(time.time() - start_time) + '\n')


def load_class():
    global hog_list
    global labels
    global loaded_flag

    with open('../data/HOG_N_LABELS/HOG_N_LABELS.pickle') as f:
        hog_tuple = pickle.load(f)
    hog_list = hog_tuple[0]
    labels = hog_tuple[1]
    loaded_flag = 1
    print ('Loaded')


def objects_detector(img_bgr8):
    global b_size
    global c_size
    global callback_rgb_timer
    global clf
    global debug_flag
    global hog_list
    global img_clean_bgr_learn
    global img_clean_gray_class
    global interactions_flag
    global labels
    global last_hog
    global live_cnt
    global live_flag
    global n_bin
    global recording_flag
    global saved_flag
    global saving_learn
    global saving_learn
    global saving_test
    global show_flag
    global shuffled_x
    global shuffled_y
    global start_time2

    objects_detector_time = rospy.get_time()
    width, height, d = np.shape(img_bgr8)
    if width > 130 or height > 130:
        return
    if width < 100 or height < 100:
        return
    # noinspection PyUnusedLocal
    detected_objects_list = []
    w, l, d = np.shape(img_bgr8)
    img_clean_bgr_learn = img_bgr8[2:w - 2, 2:l - 2].copy()
    # cv2.imshow('Learn', img_clean_bgr_learn)
    img_bgr8 = img_bgr8[7:w - 4, 9:l - 8]
    img_clean_bgr_class = img_bgr8.copy()
    img_clean_bgr_class = cv2.resize(img_clean_bgr_class, (120, 120), interpolation=cv2.INTER_AREA)  # resize image
    img_clean_gray_class = cv2.cvtColor(img_clean_bgr_class, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Clean', img_clean_bgr_class)

    if recording_flag == 1:

        learn_hog(img_clean_bgr_learn)
        # HOG_LIST.append(last_hog)
        # labels.append(LABEL)
        interactions_flag += 1
        print (interactions_flag)
        if interactions_flag == 20:
            recording_flag = 0
            interactions_flag = 0
            print ('Done recording')

    # to learn with human
    if saving_learn == 1:
        cv2.imwrite('LRN_IMGS/' + label + '_' + str(saved_flag) + '_' + color + '.png', img_clean_bgr_learn)
        saved_flag += 1
        print (saved_flag)
        if saved_flag == 3:
            saving_learn = 0
            saved_flag = 0
            print ('Done saving')

    cv2.imshow('Save_test', img_clean_bgr_class)

    # to learn with human
    if saving_test == 1:
        cv2.imwrite('TST_IMGS/' + label + '_' + str(rotation) + '_' +
                    str(saved_flag) + '_' + color + '.png', img_clean_bgr_class)
        saved_flag += 1
        print (saved_flag)
        if saved_flag == 20:
            saving_test = 0
            saved_flag = 0
            print ('Done saving')
    best_rot = 0
    best_perc = 0

    # to learn alone
    if live_flag == 1:
        start_time3 = rospy.get_time()
        if live_cnt == 0:
            start_time2 = rospy.get_time()
            live_cnt = 100
            nb_depth_img_callback(1)
            hog_list = list()
            labels = list()
            hog_list.extend(shuffled_x[1:int(len(shuffled_x) * (1.0 / len((np.unique(shuffled_y)))))])
            labels.extend(shuffled_y[1:int(len(shuffled_x) * (1.0 / len(np.unique(shuffled_y))))])
        learn_hog(img_bgr8)
        shuffled_range = range(len(labels))
        shuffled_x_temp = []
        shuffled_y_temp = []
        for i in range(5):
            random.shuffle(shuffled_range)
            shuffled_x_temp = [hog_list[i] for i in shuffled_range]
            shuffled_y_temp = [labels[i] for i in shuffled_range]
        print (live_cnt)
        live_cnt -= 1
        if live_cnt == 0:
            start_time = rospy.get_time()
            clf.partial_fit(shuffled_x_temp, shuffled_y_temp)
            print('Elapsed Time LEARNING = ' + str(rospy.get_time() - start_time) + '\n')
            live_flag = 0
            nb_depth_img_callback(NB_MAX_DEPTH_IMG)
            print('Elapsed Time TOTAL = ' + str(rospy.get_time() - start_time2)
                  + ' FPS = ' + str(100 / (rospy.get_time() - start_time2)) + '\n')
        print('Elapsed Time Single Example = ' + str(rospy.get_time() - start_time3) + '\n')
        print('Elapsed Time Single Example Obj Detc = ' + str(rospy.get_time() - objects_detector_time) + '\n')

        print('Elapsed Time Single Example RGB callback = ' + str(rospy.get_time() - callback_rgb_timer) + '\n')

    if show_flag == 0:
        return

    # fd, hog_image = hog(img_clean_GRAY_class, orientations=n_bin, pixels_per_cell=(c_size, c_size),
    #                         cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)
    # fd = np.reshape(fd, (32, 8))
    # fd_new = np.roll(fd, 2, axis=1)
    # fd_new = np.reshape(fd, (1, 32*8))[0]
    # print ('New')
    # print(fd_new[0:20])
    # print (len(fd_new))
    # rows, cols = img_clean_GRAY_class.shape
    # m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    # img_clean_bgr_class = cv2.warpAffine(img_clean_bgr_class, m, (cols, rows))
    # fd, hog_image = hog(img_clean_GRAY_class, orientations=n_bin, pixels_per_cell=(c_size, c_size),
    #                     cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)
    # print ('Rotated Original')
    # print (fd[0:20])
    # print (len(fd))

    # fd2, hog_image = hog(img_clean_GRAY_class, orientations=n_bin, pixels_per_cell=(c_size, c_size),
    #                     cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)
    # noinspection PyArgumentList
    opencv_hog = cv2.HOGDescriptor((128, 128), (b_size, b_size), (c_size, c_size), (c_size, c_size), n_bin)

    for i in range(4):
        # fd2_ori = fd2.copy()
        # fd2 = fd2.reshape(1, -1)
        # fd, hog_image = hog(img_clean_GRAY_class, orientations=n_bin, pixels_per_cell=(c_size, c_size),
        #                     cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)

        img_clean_gray_class = cv2.resize(img_clean_gray_class, (128, 128))

        fd4 = opencv_hog.compute(img_clean_gray_class)

        # fd_ori = fd.copy()
        fd4 = fd4.reshape(1, -1)
        for percentage in clf.predict_proba(fd4)[0]:
            if percentage > best_perc:
                best_perc = percentage
                best_rot = i
        # fd = fd_ori
        # fd2 = fd2_ori
        rows, cols = img_clean_gray_class.shape
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
        img_clean_gray_class = cv2.warpAffine(img_clean_gray_class, m, (cols, rows))
        # fd2 = np.reshape(fd2, (32, 8))
        # if not i == 0:
        #     fd2 = np.roll(fd2, 2, axis=1)
        # fd2 = np.reshape(fd2, (1, 32*8))[0]
        # print ('Original')
        # print (fd[0:20])
        # print ('Fake')
        # print (fd2[0:20])
    if debug_flag == 1:
        # print clf.predict(fd)
        print (best_perc)
        print ('\n')

    # print (best_rot)
    if not best_rot == 0:
        rows, cols, d = img_clean_bgr_class.shape
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), best_rot * 90, 1)
        img_clean_bgr_class = cv2.warpAffine(img_clean_bgr_class, m, (cols, rows))
    cv2.imshow('Sent', cv2.resize(img_clean_bgr_class, (256, 256)))

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


def save_hog():
    global clf
    hog_tuple = (hog_list, labels)
    # print ('Hog = ' + str(hog_tuple[0]))
    print ('labels = ' + str(np.unique(hog_tuple[1])))
    # clf.fit(hog_tuple[0], hog_tuple[1])
    with open('HOG_N_LABELS/HOG_N_LABELS.pickle', 'w') as f:
        pickle.dump(hog_tuple, f)
    # joblib.dump(clf, 'Classifier/filename.pkl')
    print ('Done')


def save_img_learn():
    global label, color, saving_learn
    mode = str(raw_input('Label: '))
    label = mode
    color_ = str(raw_input('Color: '))
    color = color_
    saving_learn = 1


def save_img_test():
    global label, color, rotation, saving_test
    mode = str(raw_input('Label: '))
    label = mode
    color_ = str(raw_input('Color: '))
    color = color_
    rotation = str(raw_input('Rotation: '))
    saving_test = 1


def show_color_img():
    if show_color_img_flag == 1:
        cv2.namedWindow("Color Img", cv2.WINDOW_NORMAL)
        cv2.imshow("Color Img", color_img)
    else:
        cv2.destroyWindow("Color Img")


def show_controls_window():
    print("Show controls' window")
    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)

    cv2.createTrackbar(RUN_NAME, MAIN_WINDOW, 0, 1, run_callback)
    cv2.createTrackbar(SHOW_COLOR_IMG, MAIN_WINDOW, 0, 1, show_color_img_callback)
    cv2.createTrackbar(SHOW_DEPTH_IMG, MAIN_WINDOW, 0, 1, show_depth_img_callback)
    cv2.createTrackbar(LEARN_FROM_DISK, MAIN_WINDOW, 0, 1, learn_from_disk_callback)
    cv2.createTrackbar(TEST_FROM_DISK, MAIN_WINDOW, 0, 1, test_from_disk_callback)
    cv2.createTrackbar(UNKNOWN_OBJECT, MAIN_WINDOW, 0, 1, unknown_object_callback)

    cv2.imshow(MAIN_WINDOW, 0)


def show_depth_img():
    if show_depth_img_flag == 1:
        cv2.namedWindow("Depth Img", cv2.WINDOW_NORMAL)
        cv2.imshow("Depth Img", depth_img)
        # cv2.imshow("Depth Img", depth_img_averaging.average())
    else:
        cv2.destroyWindow("Depth Img")


def test_from_disk():
        print ('Testing from disk')
        start_time = rospy.get_time()
        total = 0
        failure = 0
        for filename in os.listdir(TEST_PATH):
            total += 1
            rotation_num = int(filename.rsplit('_', 3)[1])
            # print ('Label ' + str(LABEL))
            # print 'Rotation ' + str(ROTATION)
            imagee = cv2.imread(TEST_PATH + filename)
            imagee = cv2.resize(imagee, (128, 128))
            found_rot = get_img_rot(imagee)
            if not abs(rotation_num - found_rot) < 0.5:
                # print ('Testing ' + str(filename))
                failure += 1
                # print ('Does not work')
                # cv2.imshow('Did not work',imagee)
                # cv2.waitKey(100)
                # print (found_rot)
                # print (ROTATION)
        percentage = 100 * failure / total
        print ('Failure = ' + str(percentage) + '%')
        print ('Failures = ' + str(failure))
        print('Elapsed Time Testing = ' + str(rospy.get_time() - start_time) + '\n')
        print ('Done')


#################
# Main function #
#################

if __name__ == '__main__':
    rospy.init_node('imageToObjects_inQuickTime', anonymous=True)

    rospy.Timer(rospy.Duration(0.02), main_callback)

    show_controls_window()

    color_buffer_msg.set_subscriber("/camera/rgb/image_rect_color")
    depth_buffer_msg.set_subscriber("/camera/depth_registered/image_raw")

    depth_buffer_msg.run()
    color_buffer_msg.run()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down")
        exit(1)
        cv2.destroyAllWindows()
