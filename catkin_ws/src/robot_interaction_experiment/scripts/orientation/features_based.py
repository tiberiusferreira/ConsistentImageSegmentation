import numpy as np
import cv2
import copy
import pca

# gets an bgr8 image and returns how many sifts are in (upright, upleft, downright, downleft) parts of the image
def sifts_location(img_bgr8, use_response):
    gray = cv2.cvtColor(img_bgr8, cv2.COLOR_BGR2GRAY)
    row, col = np.shape(gray)
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    upright = 0
    upleft = 0
    downright = 0
    downleft = 0
    for points in kp: # for each feature found
        if points.pt[1] >= col/2: # if point is in lower half of the image
            if points.pt[0] >= row/2: # if point is in right side of image
                downright += 1 + use_response*(-1 + points.response)
            else:
                downleft += 1 + use_response*(-1 + points.response)
        elif points.pt[1] < col/2:  # if point is the upper half of the image
            if points.pt[0] >= row/2:   # if the point is in right side of image
                upright += 1 + use_response*(-1 + points.response)
            else:
                upleft += 1 + use_response*(-1 + points.response)
    return upright, upleft, downright, downleft

# gets a thresholded image in bgr8 and returns how many points are in (upright, upleft, downright, downleft)
# parts of the image
def countourpnts_location(img_bgr8):
    gray = cv2.cvtColor(img_bgr8, cv2.COLOR_BGR2GRAY)
    row, col = np.shape(gray)
    upright = 0
    upleft = 0
    downright = 0
    downleft = 0
    for x in range(row):
        for y in range(col):
            if gray[x][y] > 0:
                if y >= col / 2:  # if point is in lower half of the image
                    if x >= row / 2:  # if point is in right side of image
                        downright += 1
                    else:
                        downleft += 1
                elif y < col / 2:  # if point is the upper half of the image
                    if x >= row / 2:  # if the point is in right side of image
                        upright += 1
                    else:
                        upleft += 1
    return upright, upleft, downright, downleft


# gets an bgr8 image and rotates it 180 degrees to make sure there are more sift features on its upper half
def sifts_up(img_bgr8, response):
    upright, upleft, downright, downleft = sifts_location(img_bgr8, response)
    up = (upright + upleft)
    down = (downright + downleft)
    print ('UpDown sifts ' + str(up) + " " + str(down))
    if up < down:
        img_bgr8 = pca.rotate_90(img_bgr8)
        img_bgr8 = pca.rotate_90(img_bgr8)
    return img_bgr8

# gets an bgr8 image and rotates it 180 degrees to make sure there are more sift features on its upper half
def countourpnts_up(img_bgr8):
    img_bgr8_thresh = pca.threshold_img(img_bgr8)
    upright, upleft, downright, downleft = countourpnts_location(img_bgr8_thresh)
    up = (upright + upleft)
    down = (downright + downleft)
    print ('UpDown contours ' + str(up) + " " + str(down))
    if up < down:
        img_bgr8 = pca.rotate_90(img_bgr8)
        img_bgr8 = pca.rotate_90(img_bgr8)
    return img_bgr8


def draw_sift(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)
    img_cp = img.copy()
    cv2.drawKeypoints(img, kp, img_cp)
    return img_cp
#
#     print ('\n')
#     # print (len(kp))
#     # print (up)
#     # print (down)
#     print('Up right = '+ str(upright))
#     print('Up left = ' + str(upleft))
#     print('Down right = ' + str(downright))
#     print('Down left = ' + str(downleft))
#     # if up> down:
#     #     print ('Up ' + str(up-down))
#     # else:
#     #     print ('Down ' + str(down-up))
#     # print (len(kp))
#     # print (angle/len(kp))
#     cv2.drawKeypoints(img, kp, img)
#     cv2.imshow('After', img)
#     cv2.waitKey(0)
# cv2.waitKey(0)
