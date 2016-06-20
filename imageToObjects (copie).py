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
import copy

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
indiceboxes = 0
depth_img_Avg = 0
img_bgr8_clean = 0
got_color = False
got_depth = False
average_points = np.zeros((3, 2))

def nothing(x):
    pass


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
    except CvBridgeError, e:
        print e
        return
    cleanimage = clean(img, 255)
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

    if show_depth:
        # shows the image after processing
        cv2.imshow("Depth", img)
        cv2.waitKey(1)
    else:
        cv2.destroyWindow("Depth")


def callback_rgb(msg):
    # processing of the color image
    global img_bgr8_clean, got_color
    # getting image
    try:
        img = CvBridge().imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print e
        return
    img = img[32:992, :]  # crop the image because it does not have the same aspect ratio of the depth one
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
    depth_img_Avg = cv2.resize(depth_img_Avg, (WIDTH, HEIGHT))

    # generate a mask with the closest points
    img_detection = np.where(depth_img_Avg < closest_pnt + val_depth_capture, depth_img_Avg, 0)
    # put all the pixels greater than 0 to 255
    ret, mask = cv2.threshold(img_detection, 0.0, 255, cv2.THRESH_BINARY)
    mask = np.array(mask, dtype=np.uint8)  # convert to 8-bit
    im2, contours, hierarchy = cv2.findContours(mask, 1, 2)
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

    objects_detector(uprightrect, 0)


def objects_detector(img_bgr8, i):
    width, height, d = np.shape(img_bgr8)
    if width > 130 or height > 130:
        return
    detected_objects_list = []
    img_bgr8_copy = img_bgr8.copy()
    hsv = cv2.cvtColor(img_bgr8_copy, cv2.COLOR_RGB2HSV)
    # define the values range
    hh = 255
    hl = 0
    sh = 255
    sl = 50  # filter the white color background
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
    # ret, img_gray = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)
    cv2.imshow('Filtered grayscale', img_gray)
    cv2.waitKey(1)
    maxx = 0
    img_gray_copy = img_gray.copy()
    im2, contours, hierarchy = cv2.findContours(img_gray_copy.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Find the index of the largest contour
    if not contours:
        print 'No contours found =('
        return
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    # max_len_index = np.argmax(len_)
    cnt = contours[max_index]

    # print cv2.contourArea(cnt)

    epsilon = 0.0001 * cv2.arcLength(cnt, True)
    cnt = cv2.approxPolyDP(cnt, epsilon, True)
    # for i in range(4):
    #     # find the image contours
    #     im2, contours, hierarchy = cv2.findContours(img_gray_copy.copy(), 1, cv2.CHAIN_APPROX_NONE)
    #     # Find the index of the largest contour
    #     if not contours:
    #         print 'No contours found =('
    #         return
    #     areas = [cv2.contourArea(c) for c in contours]
    #     max_index = np.argmax(areas)
    #     cnt = contours[max_index]
    #     x, y, width, height = cv2.boundingRect(cnt)
    #     cv2.min
    #     hull = cv2.convexHull(cnt)
    #
    #     Mom = cv2.moments(hull)
    #     cx = int(Mom['m10'] / Mom['m00'])**2
    #     cy = int(Mom['m01'] / Mom['m00'])**2
    #     summ = cx-cy
    #     print str(i) + ' Cx = ' + str(cx)
    #     print str(i) + ' Cy = ' + str(cy)
    #     print str(i) + ' Sum = ' + str(cx-cy)
    #     if maxx < summ:
    #         maxx = summ
    #         final_cont = cnt
    #         img_gray = img_gray_copy
    #     rows, cols = img_gray_copy.shape
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    #     img_gray_copy = cv2.warpAffine(img_gray_copy, M, (cols, rows))
    #     print '\n'
    # cv2.imshow('GOLDEN!',img_gray)
    #


    # cv2.circle(contour_img_cropped, (cx, cy), 3, 255, -1)
    # Moments = cv2.moments(cnt)
    # cx = int(Moments['m10'] / Moments['m00'])
    # cy = int(Moments['m01'] / Moments['m00'])


    # Mom = cv2.moments(cnt)
    # cx = int(Mom['m10'] / Mom['m00'])
    # cy = int(Mom['m01'] / Mom['m00'])
    # summ = cx + cy
    # print str(i) + ' Cx = ' + str(cx)
    # print str(i) + ' Cy = ' + str(cy)
    # print str(i) + ' Sum = ' + str(cx+cy)

    # ellipse = cv2.fitEllipse(cnt)
    # cv2.ellipse(contour_img, ellipse, (0, 255, 0), 2)
    hull = cv2.convexHull(cnt)
    # epsilon = 0.001 * cv2.arcLength(cnt, True)
    # cnt = cv2.approxPolyDP(cnt, epsilon, True)
    height, width, channels = img_bgr8_copy.shape
    contour_img = img_bgr8_copy.copy()
    cv2.drawContours(contour_img, cnt, -1, (0, 255, 0), 3)
    points = cv2.minEnclosingTriangle(hull)
    # print points
    points00 = points[1][0][0][0]
    points01 = points[1][0][0][1]
    points10 = points[1][1][0][0]
    points11 = points[1][1][0][1]
    points20 = points[1][2][0][0]
    points21 = points[1][2][0][1]
    a_new = np.array([(points00, points01), (points10, points11), (points20, points21)])
    a_new_new = a_new.copy()
    a_old = average_points
    print 'a_new'
    print a_new
    print 'a_old'
    print a_old
    # print len(a_old)
    for i2 in range(3):
        indice_min = -1
        dist_deb = 99999999
        for i in range(3):  # for each point in a_new compare with the first point in the vector we have
            dist = np.linalg.norm(a_new[i]-a_old[i2])
            if dist < dist_deb:
                dist_deb = dist
                indice_min = i
        #     print 'dist ' + str(i)
        #     print dist
        # print indice_min
        a_new_new[i2] = a_new[indice_min].copy()
        a_new[indice_min] = 9999999
    print 'A new new'
    print a_new_new


    global pointsIndex
    global lastpoints
    pointsIndex += 1
    global NUMBER_LAST_POINTS
    if pointsIndex >= NUMBER_LAST_POINTS:
        pointsIndex = 0
    lastpoints[pointsIndex] = a_new_new.copy()
    print 'All last!'
    # print lastpoints
    # print lastpoints
    #
    # # print lastpoints[pointsIndex]
    # print 'Points'

    global average_points
    # avg_point = np.array([(0, 0), (0, 0), (0, 0)])
    for i in range(NUMBER_LAST_POINTS):
        average_points = average_points + lastpoints[i]
    print 'avg'
    average_points = average_points/(NUMBER_LAST_POINTS+1)
    # print avg_point

    #
    # Mom = cv2.moments(hull)
    # print Mom
    # cx = int(Mom['m10'] / Mom['m00'])
    # cy = int(Mom['m01'] / Mom['m00'])
    # # summ = cx + cy
    # print str(i) + ' Cx = ' + str(cx)
    # print str(i) + ' Cy = ' + str(cy)
    # print str(i) + ' Sum = ' + str(cx+cy)

    # avg_point = avg_point/3
    # print 'Avg = '
    # print avg_point
    # print avg_point
    # print avg_point

    # print type(avg_point)

    # lastpoints[pointsIndex]
    # global pointsIndex
    # global lastpoints
    # # print points
    # if pointsIndex > NB_DEPTH_IMGS:
    #     pointsIndex = 0
    # lastpoints[pointsIndex] = np.copy(points)
    # pointsIndex += 1
    # # creates an image which is the average of the last ones
    # points_avg = np.copy(lastpoints[0])
    # # cnt_avg = np.array(0)
    # for i in range(1, NB_DEPTH_IMGS):
    #     points_avg += lastpoints[i]
    # print NB_DEPTH_IMGS
    # points_avg /= NB_DEPTH_IMGS
    # points = points_avg

    p0 = average_points[0]
    p1 = average_points[1]
    p2 = average_points[2]
    dist0 = np.linalg.norm(p0 - p1)
    dist1 = np.linalg.norm(p1 - p2)
    dist2 = np.linalg.norm(p0 - p2)
    print 'Dist 0'
    print dist0
    print 'Dist 1'
    print dist1
    print 'Dist 2'
    print dist2
    dist_arr = [dist0, dist1, dist2]
    maxx = np.argmax(dist_arr)
    dist_arr2 = copy.copy(dist_arr)
    dist_arr2[maxx] = -9999
    print dist_arr
    print dist_arr2
    maxx2 = np.argmax(dist_arr2)
    print 'Abs = '
    print abs(dist_arr[maxx]-dist_arr[maxx2])
    max_abs = abs(dist_arr[maxx]-dist_arr[maxx2])

    minn = np.argmin(dist_arr)
    dist_arr2 = copy.copy(dist_arr)
    dist_arr2[minn] = 99999
    # print dist_arr
    # print dist_arr2
    minn2 = np.argmin(dist_arr2)
    print 'Abs = '
    print abs(dist_arr[minn]-dist_arr[minn2])
    min_abs = abs(dist_arr[minn]-dist_arr[minn2])

    if min_abs > max_abs*3:
        maxx = np.argmin(dist_arr)
    elif max_abs > min_abs*3:
        maxx = np.argmax(dist_arr)
    else:
        maxx = np.argmin(dist_arr)

    # print minn
    if maxx == 0:
        # print p0
        # print p1
        point_zica = [p0, p1]
        dist_arr[0] = 0
    if maxx == 1:
        # print p1
        # print p2
        point_zica = [p1, p2]
        dist_arr[1] = 0
    if maxx == 2:
        # print p0
        # print p2
        point_zica = [p0, p2]
        dist_arr[2] = 0

    # maxx = np.argmax(dist_arr)
    # if maxx == 0:
    #     # print p0
    #     # print p1
    #     point_zica2 = [p0, p1]
    #     dist_arr[0] = 0
    # if maxx == 1:
    #     # print p1
    #     # print p2
    #     point_zica2 = [p1, p2]
    #     dist_arr[1] = 0
    # if maxx == 2:
    #     # print p0
    #     # print p2
    #     point_zica2 = [p0, p2]
    #     dist_arr[2] = 0

    # print '\n'
    # print 'Dist0 = ' + str(dist0)
    # print 'Dist1 = ' + str(dist1)
    # print 'Dist2 = ' + str(dist2)
    # cv2.polylines(contour_img, hull, True, (255, 0, 0), 3)
    # cv2.drawContours(contour_img,hull,-1,(255,0,0))
    cv2.polylines(contour_img, np.int32([hull]), True, 255)
    # cv2.polylines(contour_img, np.int32([points[1]]), True, 255)
    contour_img = cv2.copyMakeBorder(contour_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT)
    # average_points += 30
    cv2.drawContours(contour_img, np.int32([point_zica]), -1, (0,255,0), offset=(30,30))
    cv2.drawContours(contour_img, np.int32([points[1]]), -1, (255,0,0), offset=(30,30))
    cv2.drawContours(contour_img, np.int32([average_points]), -1, (0,0,255), offset=(30,30))

    cv2.imshow('HULL', contour_img)
    # Mom = cv2.moments(hull)
    # cx = int(Mom['m10'] / Mom['m00'])
    # cy = int(Mom['m01'] / Mom['m00'])
    # summ = cx + cy
    # print str(i) + ' Cx = ' + str(cx)
    # print str(i) + ' Cy = ' + str(cy)
    # print str(i) + ' Sum = ' + str(summ)
    # ellipse = cv2.fitEllipse(cnt)
    # cv2.ellipse(contour_img, ellipse, (0, 255, 0), 2)
    # get rotated rect of contour and split into components
    # center, size, angle = cv2.minAreaRect(cnt)
    # box = cv2.cv.BoxPoints(cv2.minAreaRect(cnt))
    # box = np.int32(box)
    x, y, width, height = cv2.boundingRect(cnt)
    contour_img_box = contour_img.copy()
    # cv2.rectangle(contour_img_box, (x, y), (x + width, y + height), (0, 255, 0), 2)
    # cv2.drawContours(contour_img, [box], 0, (0, 0, 255), 2)
    # cv2.drawContours(contour_img, [box2], 0, (0, 0, 255), 2)
    cv2.imshow('Contour', contour_img_box)
    cv2.waitKey(1)
    cropped_bgr8 = img_bgr8_copy[y:y + height, x:x + width]
    contour_img_cropped = contour_img[y:y + height, x:x + width]
    cv2.imshow('zica', cropped_bgr8)
    # if height > 1.2*width:
    #     rotmat = cv2.getRotationMatrix2D((height / 2.0, width / 2.0), 90, 1.0)
    #     roi = cv2.warpAffine(roi, rotmat, (height, width), flags=cv2.INTER_LINEAR)  # INTER_CUBIC
    # cv2.imshow('uprightRect', roi)
    # std_length = 80
    n_bin = 4  # number of orientations for the HoG
    b_size = 2  # block size
    c_size = 2  # cell size
    cropped_gray = cv2.cvtColor(cropped_bgr8, cv2.COLOR_BGR2GRAY)
    cv2.imshow('CropGray', cropped_gray)
    fd, hog_image = hog(cropped_gray, orientations=n_bin, pixels_per_cell=(c_size, c_size),
                        cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 4))
    # cv2.imshow('HOGG', hog_image)
    features_hog = fd
    features_hog = np.reshape(features_hog, (np.shape(features_hog)[0]/4, 4))
    sum_features = sum(features_hog)
    # print np.shape(features_hog)
    # print sum_features
    # # print np.sum(sum_features)
    # print np.argmax(sum_features)
    # rows, cols = contour_img_cropped.shape[:2]
    # [vx, vy, x, y] = cv2.fitLine(cnt,cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
    # lefty = int((-x * vy / vx) + y)
    # righty = int(((cols - x) * vy / vx) + y)
    # cv2.line(contour_img_cropped, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
    height, width = cropped_gray.shape
    cv2.imshow('CropGray', cropped_gray)
    # cropped_gray_thres = np.where(cropped_gray > 170, cropped_gray, 0)
    # cv2.imshow('CropThre',cropped_gray_thres)
    # print blank
    cv2.imshow('AAA', contour_img_cropped)
    contour_img_cropped = np.where(contour_img_cropped[:, :, 1] == 255, contour_img_cropped[:, :, 1], 0)
    cv2.imshow('BBBB', contour_img_cropped)
    sum1 = list()
    sum1.append(contour_img_cropped[0:height / 2, 0:width / 2])
    sum1.append(contour_img_cropped[height / 2:height, 0:width / 2])
    sum1.append(contour_img_cropped[0:height / 2, width / 2:width])
    sum1.append(contour_img_cropped[height / 2:height, width / 2:width])
    # cv2.imshow('a', sum1[0])
    # cv2.imshow('b', sum1[1])
    # cv2.imshow('c', sum1[2])
    # cv2.imshow('d', sum1[3])
    sum_tot = [0, 0, 0, 0]
    #
    # for i in range(4):
    #     # std_length = 80
    #     n_bin = 4  # number of orientations for the HoG
    #     b_size = 64  # block size
    #     c_size = 8  # cell size
    #     # cropped_cnt_gray = cv2.cvtColor(contour_img_cropped, cv2.COLOR_BGR2GRAY)
    #     fd, hog_image = hog(contour_img_cropped, orientations=n_bin, pixels_per_cell=(c_size, c_size),
    #                         cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)
    #     hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 4))
    #     # cv2.imshow('HOGG', hog_image)
    #     features_hog = fd
    #     features_hog = np.reshape(features_hog, (np.shape(features_hog)[0]/4, 4))
    #     sum_features = sum(features_hog)
    #     # print np.shape(features_hog)
    #     print i
    #     # print min(sum_features)
    #     print sum_features
        # print np.sum(sum_features)
        # print np.argmax(sum_features)
    #     sum1 = list()
    #     sum1.append(contour_img_cropped[0:height / 2, 0:width / 2])
    #     sum1.append(contour_img_cropped[height / 2:height, 0:width / 2])
    #     sum1.append(contour_img_cropped[0:height / 2, width / 2:width])
    #     sum1.append(contour_img_cropped[height / 2:height, width / 2:width])
    #     sum_index = 0
    #     highest_first = 0
    #     highest_first_index = 0
    #     summ = 0
    #     # for sum_ in sum1:
    #     summ = 0
    #     height, width = contour_img_cropped.shape
    #     for index_lin, lin in enumerate(contour_img_cropped):
    #         # print lin
    #         # print index_lin
    #         index_col = 0
    #         for col in lin:
    #             if col == 255:
    #                 summ = summ + index_col ** 2
    #             index_col += 1
    #             # print summ
    #             # sum_lin = sum_[:, :]
    #             # first_255 = 0
    #             # for index, point in enumerate(sum_lin):
    #             #     if point == 255:
    #             #         sum_tot[sum_index] = sum_tot[sum_index] + index
    #             #         first_255 = index
    #             #
    #             # print ' First = ' + str(first_255)
    #             # if highest_first < first_255:
    #             #     highest_first = first_255
    #             #     highest_first_index = sum_index
    # #             # sum_index += 1
    #     rows, cols = contour_img_cropped.shape
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    #     contour_img_cropped = cv2.warpAffine(contour_img_cropped, M, (cols, rows))
    # #     # print 'Stage ' + str(i)
    #     sum_tot[i] = summ
    # print sum_tot
    # print sum_tot.index((min(sum_tot)))
    #
    # rows, cols, d = cropped_bgr8.shape
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), sum_tot.index((min(sum_tot))) * 90, 1)
    # cropped_bgr8 = cv2.warpAffine(cropped_bgr8, M, (cols, rows))
    # print cv2.
    cv2.imshow('Sent', cv2.resize(cropped_bgr8, (256, 256)))
    # if not highest_first_index == 0 and i < 5:
    #     rows, cols, d = img_bgr8.shape
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    #     img_bgr8 = cv2.warpAffine(img_bgr8, M, (cols, rows))
    #     objects_detector(img_bgr8, i+1)
    #     return


    # cv2.imshow('KKKK', sum_lin)
    # n_bin = 4  # number of orientations for the HoG
    # b_size = 8  # block size
    # c_size = 8  # cell size
    # fd, hog_image = hog(sum_, orientations=n_bin, pixels_per_cell=(c_size, c_size),
    #                     cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)
    # features_hog = fd
    # features_hog = np.reshape(features_hog, (np.shape(features_hog)[0] / 4, 4))
    # sum_features = sum(features_hog)
    # print 'Sum1 =' + str(sum_features)
    # print '\n'
    # detected_object = Detected_Object()
    # detected_object.id = 1
    # detected_object.image = CvBridge().cv2_to_imgmsg(cv2.resize(cropped_bgr8, (256, 256), interpolation=cv2.INTER_AREA),
    #                                                  encoding="bgr8")
    # detected_object.center_x = 0  # unrot_center_x / float(resolution_x)  # proportion de la largeur
    # detected_object.center_y = 0  # unrot_center_y / float(resolution_x)  # proportion de la largeur aussi
    # detected_object.features = getpixelfeatures(cropped_bgr8)
    # detected_object.features.hog_histogram = GetHOGFeatures(cropped_bgr8)
    # detected_objects_list.append(detected_object)
    # detected_objects_list_msg = Detected_Objects_List()
    # detected_objects_list_msg.detected_objects_list = detected_objects_list
    # detected_objects_list_publisher.publish(detected_objects_list_msg)
    # cv2.imshow('Sent', cv2.resize(cropped_bgr8, (256, 256)))



    # if np.argmax(sum_features) == 0:
    #     rows, cols = cropped.shape
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    #     cropped = cv2.warpAffine(cropped, M, (cols, rows))
    # cv2.imshow('AfterHOG ROT', cropped)



    # # if we rotate more than 90 degrees, the width becomes height and vice-versa
    # if angle < -45.0:
    #     angle += 90.0
    #     width, height = size[0], size[1]
    #     size = (height, width)
    # rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # # rotate the entire image around the center of the parking cell by the
    # # angle of the rotated rect
    # imgwidth, imgheight = (img_bgr8_copy.shape[0], img_bgr8_copy.shape[1])
    # rotated = cv2.warpAffine(img_bgr8_copy, rot_matrix, (imgheight, imgwidth), flags=cv2.INTER_CUBIC)
    # # extract the rect after rotation has been done
    # sizeint = (np.int32(size[0]), np.int32(size[1]))
    # uprightrect = cv2.getRectSubPix(rotated, sizeint, center)
    # height, width, channels = uprightrect.shape
    # # if height > 1.2*width:
    # #     rotmat = cv2.getRotationMatrix2D((height / 2.0, width / 2.0), 90, 1.0)
    # #     uprightrect = cv2.warpAffine(uprightrect, rotmat, (height, width), flags=cv2.INTER_LINEAR)  # INTER_CUBIC
    # cv2.imshow('uprightRect', uprightrect)



    # img_bgr8_cropped_rotated = cv2.getRectSubPix(img_bgr8_rotated, (unrot_h, unrot_w), ((unrot_x+unrot_h/2),
    #                                                                                     (unrot_y+unrot_w/2)))


    # treat each contour
    # for cnt in contours:
    #     min_area_rect = cv2.minAreaRect(cnt)  # minimum area rectangle that encloses the contour cnt
    #     box = cv2.cv.BoxPoints(min_area_rect)  # returns the four vertices of the rotated rectangle
    #     box = np.int32(box)
    #     unrot_x, unrot_y, unrot_w, unrot_h = cv2.boundingRect(np.array([box]))  # returns the points of an upright box
    #     # the idea here is to check if the objects bonding box is close to the last ones, if so the image is stable,
    #     # the object well segmented and we take its "picture"
    #     global indiceboxes
    #     global lastBoxes
    #     global NB_INDEX_BOX
    #     # only save up to a certain number of last bonding boxes
    #     if indiceboxes >= NB_INDEX_BOX:
    #         indiceboxes = 0
    #     lastBoxes[indiceboxes] = np.copy(cv2.boundingRect(np.array([box])))
    #     meanboxes = 0  # mean value of the last boxes
    #     meanall = range(NB_INDEX_BOX)
    #     for i in range(0, NB_INDEX_BOX):
    #         meanboxes += lastBoxes[i]
    #         meanall[i] += np.sum(lastBoxes[i])
    #         meanall[i] /= NB_INDEX_BOX
    #     meanboxes /= NB_INDEX_BOX
    #     indiceboxes += 1
    #     img_bgr8_resized = cv2.resize(img_bgr8, (objects_image_size, objects_image_size))
    #     # HISTOGRAMS
    #     object_img_rgb = cv2.cvtColor(img_bgr8_resized, cv2.COLOR_BGR2RGB)
    #     object_img_rgb2 = object_img_rgb.copy()
    #     rotated_img_obj = np.copy(object_img_rgb)


    #
    #
    # # std_length = 80
    # n_bin = 4  # number of orientations for the HoG
    # b_size = 64  # block size
    # c_size = 8  # cell size
    # rotated_img_obj = cv2.resize(rotated_img_obj, (256, 256), interpolation=cv2.INTER_AREA)  # resize image
    # object_img_rgb2 = cv2.resize(object_img_rgb2, (256, 256), interpolation=cv2.INTER_AREA)  # resize image
    # # rotated_img_obj = rotated_img_obj[20:236, 20:236]   # cut parts that are usually not the desired image
    # # object_img_rgb2 = object_img_rgb2[20:236, 20:236]   # cut parts that are usually not the desired image
    # rotated_img_obj = rotated_img_obj[:, :, 1]  # make if black and white for the hog
    # # rotated_img_obj = np.where(rotated_img_obj[:, :] < 170, rotated_img_obj[:, :], 255)
    # # rotated_img_obj = np.where(rotated_img_obj[:, :] > 170, rotated_img_obj[:, :], 0)
    # contours2, hierarchy = cv2.findContours(rotated_img_obj.copy(), 1, 2)
    # contours2 = [cnt for cnt in contours2 if cv2.contourArea(cnt) > MIN_AREA/2]
    # # for cnt3 in contours2:
    # #     epsilon = 0.001 * cv2.arcLength(cnt3, True)
    # #     contours3.append(cv2.approxPolyDP(cnt3, epsilon, True))
    #
    # # for contt in contours2:
    # cv2.drawContours(object_img_rgb2, contours2, -1, (0, 255, 0), 3)  # Draws the rotated rectangle enclosing the contour
    # cv2.imshow('ORIGINAL', rotated_img_obj)
    # fd, hog_image = hog(rotated_img_obj, orientations=n_bin, pixels_per_cell=(c_size, c_size),
    #                     cells_per_block=(b_size / c_size, b_size / c_size), visualise=True)
    # hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 4))
    # cv2.imshow('HOGG', hog_image)
    # features_hog = fd
    # features_hog = np.reshape(features_hog, (np.shape(features_hog)[0]/4, 4))
    # sum_features = sum(features_hog)
    # print np.shape(features_hog)
    # print sum_features
    # print np.sum(sum_features)
    # print np.argmax(sum_features)
    # if np.argmax(sum_features) == 0:
    #     rows, cols = rotated_img_obj.shape
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    #     rotated_img_obj = cv2.warpAffine(rotated_img_obj, M, (cols, rows))
    # cv2.imshow('AfterHOG ROT', rotated_img_obj)
    #
    #
    #     # if the mean of the first half is greater than the second, rotate 180 degrees
    # if (np.sum(rotated_img_obj[:256 / 2]) > np.sum(rotated_img_obj[256 / 2:256])) and\
    #                 np.abs(np.sum(rotated_img_obj[:256 / 2])-np.sum(rotated_img_obj[256 / 2:256])) > 0.3*np.sum(rotated_img_obj[256 / 2:256]):
    #     rows, cols = rotated_img_obj.shape
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 2 * 90, 1)
    #     rotated_img_obj = cv2.warpAffine(rotated_img_obj, M, (cols, rows))
    #     print 'yes'
    # else:
    #     print 'no'
    # print 'Sum first half'
    # print np.sum(rotated_img_obj[:256 / 2])
    # print 'Sum second half'
    # print np.sum(rotated_img_obj[256 / 2:256])
    # rotated_img_obj = cv2.cvtColor(rotated_img_obj, cv2.COLOR_GRAY2RGB)
    #
    # #
    # rows, cols = rotated_img_obj.shape
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), max_direct*90, 1)
    # rotated_img_obj = cv2.warpAffine(rotated_img_obj, M, (cols, rows))


    #
    #
    #
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


def GetHOGFeatures(object_img_rgb):
    std_length = 80
    n_bin = 9  # number of orientations for the HoG
    b_size = 8
    b_stride = 8  # 5
    c_size = 8
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
    print "Creating windows"
    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(TRACKBAR_NB_PROFONDEUR_NAME, MAIN_WINDOW_NAME, NB_DEPTH_IMGS, NB_IMG_PROFONDEUR_MAX,
                       changeprofondeur)
    cv2.createTrackbar(AFFICHAGE_COULEUR, MAIN_WINDOW_NAME, 0, 1, changeaffcouleur)
    cv2.createTrackbar(AFFICHAGE_PROFONDEUR, MAIN_WINDOW_NAME, 0, 1, changeaffprofondeur)
    cv2.createTrackbar(CAPTURE_PROFONDEUR, MAIN_WINDOW_NAME, int(100 * val_depth_capture), 150, changecapture)
    cv2.imshow(MAIN_WINDOW_NAME, 0)
    print ("Creating subscribers")
    image_sub_rgb = rospy.Subscriber("/camera/rgb/image_rect_color", Image, callback_rgb, queue_size=50)
    image_sub_depth = rospy.Subscriber("/camera/depth_registered/image_raw/", Image, callback_depth, queue_size=50)
    detected_objects_list_publisher = rospy.Publisher('detected_objects_list', Detected_Objects_List, queue_size=10)
    print ("Spinning ROS")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
        cv2.destroyAllWindows()
