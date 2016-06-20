###########
# Imports #
###########

import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import cv2

import saveImages

#############
# Constants #
#############

DEPTH_MAX_VALUE = 10
INIT_DEPTH_MAX = DEPTH_MAX_VALUE

MAIN_WINDOW_NAME = "Controls"

IMG_FROM_FILE = "Get img from file"
DEPTH_MAX = "Change depth max"
SAVE = "Save images"

SHOW_IMG_COLOR = "Show color img"
SHOW_IMG_DEPTH = "Show depth img"

####################
# Global variables #
####################
imgFromFile = 0

showColorImg = 0
showDepthImg = 0

depthMax = INIT_DEPTH_MAX
lastSave = 0

depthImg = 0
depthImgTransform = 0
UnrefinedDepthImg = 0

colorImg = 0


#############
# Callbacks #
#############

# __NOTE__ : important function
def imgFromFileCallback(value):
    global imgFromFile
    if value == 1:
        saveImages.initialisation_file_reading()
    imgFromFile = value


# __NOTE__ : important function
def depthMaxCallback(value):
    global depthMax
    depthMax = value


# __NOTE__ : important function
def saveCallback(value):
    global lastSave
    if (lastSave == 0 and value == 1):
        saveImages.initialisation_img_saving()
        saveImages.save_color_img(colorImg)
        saveImages.save_depth_img(UnrefinedDepthImg, depthMax)
    lastSave = value


def showColorImgCallback(value):
    global showColorImg
    showColorImg = value


def showDepthImgCallback(value):
    global showDepthImg
    showDepthImg = value


def DepthImageCallback(msg):
    global depthImg, UnrefinedDepthImg
    # getting the image
    if (imgFromFile == 1):  # __NOTE__ : important function
        UnrefinedDepthImg = saveImages.read_depth_img()
    else:
        try:
            UnrefinedDepthImg = CvBridge().imgmsg_to_cv2(msg, "passthrough")
        except CvBridgeError as e:
            print (e)
            return

    depthImg = clean(UnrefinedDepthImg, 255)

    depthImg = transformDepthImg(depthImg, 255)

    if (showDepthImg == 1):
        # shows the image after processing
        cv2.imshow("Depth Image", toDisplay(depthImg))
    else:
        cv2.destroyWindow("Depth Image")
    cv2.waitKey(1)


def callback_rgb(msg):
    # processing of the color image
    global img_bgr8_clean, got_color, colorImg
    if imgFromFile == 1:  # else done by DepthImageCallback
        colorImg = saveImages.read_color_img()
    else:
        try:
            colorImg = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print (e)
            return
        # img = cv2.resize(img, (WIDTH, HEIGHT))
        # print np.shape(img)

        img = colorImg[32:992, 0:1280]  # crop the image because it does not have the same aspect ratio of the depth one
        img_bgr8_clean = np.copy(img)

    got_color = True  # ensures there is an color image available
    if (showColorImg == 1):
        # show image obtained
        cv2.imshow('couleur', colorImg)
        cv2.waitKey(1)
    else:
        cv2.destroyWindow('couleur')


#############
# Functions #
#############

def clean(img, n):
    global depthMax
    # set the non-finite values (NaN, inf) to n
    # returns 1 where the img is finite and 0 where it is not
    mask = np.isfinite(img)
    #  where mask puts img, else puts n, so where is finite puts img, else puts n
    img = np.where(mask, img, n)
    img = np.where(img < depthMax, img, n)
    return img


def transformDepthImg(img, n):
    global depthMax

    img = img * n / depthMax
    img = np.where(img <= n, img, n)
    img.astype(int)
    return img


def toDisplay(dImg):
    dImg = dImg / 255
    return dImg


#################
# Main function #
#################

if __name__ == '__main__':
    print("Init ros node")
    rospy.init_node('saveDepth', anonymous=True)

    print ("Creating windows")
    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(IMG_FROM_FILE, MAIN_WINDOW_NAME, 0, 1, imgFromFileCallback)
    cv2.createTrackbar(DEPTH_MAX, MAIN_WINDOW_NAME, INIT_DEPTH_MAX, 10, depthMaxCallback)
    cv2.createTrackbar(SAVE, MAIN_WINDOW_NAME, 0, 1, saveCallback)
    cv2.createTrackbar(SHOW_IMG_COLOR, MAIN_WINDOW_NAME, 0, 1, showColorImgCallback)
    cv2.createTrackbar(SHOW_IMG_DEPTH, MAIN_WINDOW_NAME, 0, 1, showDepthImgCallback)
    cv2.imshow(MAIN_WINDOW_NAME, 0)
    cv2.waitKey(1)

    image_sub_depth = rospy.Subscriber("/camera/depth_registered/image_raw/", Image, DepthImageCallback, queue_size=1)
    image_sub_rgb = rospy.Subscriber("/camera/rgb/image_rect_color", Image, callback_rgb, queue_size=1)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print ("Shutting down")
        cv2.destroyAllWindows()
