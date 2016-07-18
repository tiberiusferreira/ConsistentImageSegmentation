import numpy as np
import cv2
import copy

average_points = np.zeros((3, 2))
pointsIndex = 0
NUMBER_LAST_POINTS = 15
lastpoints = np.zeros((NUMBER_LAST_POINTS, 3, 2))




def objects_detector(uprightrects_tuples):
    '''Using only one image to make it easier to visualize and minimize output'''
    for index, curr_tuple in enumerate(uprightrects_tuples):
        img_bgr8, center = curr_tuple
        w, l, d = np.shape(img_bgr8)
        img_bgr8_copy = img_bgr8.copy()
    hsv = cv2.cvtColor(img_bgr8_copy, cv2.COLOR_RGB2HSV)
    # define the values range
    hh = 255
    hl = 0
    sh = 255
    sl = 100  # filter the white color background
    vh = 255
    vl = 0  # to ignore the black in the background
    lowerbound = np.array([hl, sl, vl], np.uint8)
    upperbound = np.array([hh, sh, vh], np.uint8)
    # filter the image to generate the mask
    filtered_hsv = cv2.inRange(hsv, lowerbound, upperbound)
    filtered_hsv = cv2.bitwise_and(hsv, hsv, mask=filtered_hsv)
    filtered_hsv_s = cv2.resize(filtered_hsv, (256, 256))
    cv2.imshow('Filtered', filtered_hsv_s)
    cv2.waitKey(1)
    # convert the image to grayscale in order to find contours
    img_bgr = cv2.cvtColor(filtered_hsv, cv2.COLOR_HSV2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    img_gray_af = cv2.dilate(img_gray.copy(), kernel, iterations=1)
    img_gray_s = cv2.resize(img_gray, (256, 256))
    img_gray_bf = cv2.resize(img_gray_af, (256, 256))
    cv2.imshow('FIlGRAY', img_gray_bf)
    cv2.waitKey(1)
    img_gray = img_gray_af
    cv2.imshow('Filtered grayscale', img_gray_s)
    cv2.waitKey(1)
    img_gray_copy = img_gray.copy()
    im2, contours, hierarchy = cv2.findContours(img_gray_copy.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Find the index of the largest contour
    if not contours:
        print 'No contours found =('
        return
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    epsilon = 0.0001 * cv2.arcLength(cnt, True)
    cnt = cv2.approxPolyDP(cnt, epsilon, True)
    '''Here the ideia is to rotate the image in order to always have sum(cx-cy) the lowest
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
    #     if lowest < summ:
    #         lowest = summ
    #         final_cont = cnt
    #         img_gray = img_gray_copy
    #     rows, cols = img_gray_copy.shape
    #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    #     img_gray_copy = cv2.warpAffine(img_gray_copy, M, (cols, rows))
    #     print '\n'
    # cv2.imshow('GOLDEN!',img_gray)
    #'''


    # ellipse = cv2.fitEllipse(cnt)
    # cv2.ellipse(contour_img, ellipse, (0, 255, 0), 2)
    hull = cv2.convexHull(cnt)
    # epsilon = 0.001 * cv2.arcLength(cnt, True)
    # cnt = cv2.approxPolyDP(cnt, epsilon, True)
    contour_img = img_bgr8_copy.copy()
    contour_img_clean = img_bgr8_copy.copy()
    cv2.drawContours(contour_img, cnt, -1, (0, 255, 0), 3)
    cv2.drawContours(contour_img_clean, cnt, -1, (0, 255, 0), 3)

    points = cv2.minEnclosingTriangle(hull)
    # Getting triangle x y coordinates
    points00 = points[1][0][0][0]
    points01 = points[1][0][0][1]
    points10 = points[1][1][0][0]
    points11 = points[1][1][0][1]
    points20 = points[1][2][0][0]
    points21 = points[1][2][0][1]
    a_new = np.array([(points00, points01), (points10, points11), (points20, points21)])
    a_new_new = a_new.copy()
    a_old = average_points
    print 'new points'
    print a_new
    print 'old points'
    print a_old
    # print len(a_old)
    ''' Ordering the triangle points so we can compare with the previous recored points because they come in any order.'''
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
    ''' Saving current points in the average '''
    global pointsIndex
    global lastpoints
    pointsIndex += 1
    global NUMBER_LAST_POINTS
    if pointsIndex >= NUMBER_LAST_POINTS:
        pointsIndex = 0
    lastpoints[pointsIndex] = a_new_new.copy()
    print 'All last!'
    global average_points
    # avg_point = np.array([(0, 0), (0, 0), (0, 0)])
    for i in range(NUMBER_LAST_POINTS):
        average_points = average_points + lastpoints[i]
    print 'avg'
    average_points = average_points/(NUMBER_LAST_POINTS+1)

    ''' Getting either the shortest triangle side or longest (which ever makes it most different from the other two)'''
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
    # average_points += 30
    contour_img = cv2.copyMakeBorder(contour_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT)
    '''Drawing the most different side which would be used to get a good orientation'''
    cv2.drawContours(contour_img, np.int32([point_zica]), -1, (0,255,0), offset=(30,30))
    cv2.drawContours(contour_img, np.int32([points[1]]), -1, (255,0,0), offset=(30,30))
    cv2.drawContours(contour_img, np.int32([average_points]), -1, (0,0,255), offset=(30,30))
    x, y, width, height = cv2.boundingRect(cnt)
    contour_img_box = contour_img.copy()
    resized_cnt = cv2.resize(contour_img_box, (256, 256))
    cv2.imshow('Contour', resized_cnt)
    cv2.waitKey(1)
    cropped_bgr8 = img_bgr8_copy[y:y + height, x:x + width]
    contour_img_cropped = contour_img
    resized_cnt = cv2.resize(cropped_bgr8, (256, 256))

    cv2.imshow('Final', resized_cnt)
    cropped_gray = cv2.cvtColor(cropped_bgr8, cv2.COLOR_BGR2GRAY)
    resized_cnt = cv2.resize(cropped_gray, (256, 256))


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
    cropped_gray = cv2.copyMakeBorder(cropped_gray, 30, 30, 30, 30, cv2.BORDER_CONSTANT)
    resized_cnt = cv2.resize(cropped_gray, (256, 256))

    cv2.imshow('CropGray', resized_cnt)
    # cropped_gray_thres = np.where(cropped_gray > 170, cropped_gray, 0)
    # cv2.imshow('CropThre',cropped_gray_thres)
    # print blank
    resized_cnt = cv2.resize(contour_img_cropped, (256, 256))

    contour_img_cropped = np.where(contour_img_clean[:, :, 1] == 255, contour_img_clean[:, :, 1], 0)
    resized_cnt = cv2.resize(contour_img_cropped, (256, 256))

    cv2.imshow('OnlyContour', resized_cnt)

