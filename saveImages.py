# coding=utf-8

"""
Created by François BIDET, the 15/06/16

Module for save and read images from OpenCV

6 functions:
    - initialisation_file_reading()

    - read_color_img()

    - read_depth_img()

    - initialisation_img_saving()

    - save_color_img(img)

    - save_depth_img(img, depth)
"""

###########
# Imports #
###########

import os.path  # exist file or directory

import numpy as np

import cv2

import readline, glob

#################
# Configuration #
#################

def complete(text, state):
    return (glob.glob(text+'*')+[None])[state]

readline.set_completer_delims(' \t\n;')
readline.parse_and_bind("tab: complete")
readline.set_completer(complete)

#########
# Class #
#########

class FileManager:
    def __init__(self):
        self.__file_path = ''
        self.__file_name = ''
        self.__file_suffix = 1

        self.__load_color_from_file = 0
        self.__load_depth_from_file = 0

        self.__color_img = np.array([])
        self.__depth_img = np.array([])

    def __exist_file(self):
        return os.path.exists(self.__file_path + self.__file_name + ".png")

    def __get_file_name_for_read(self, for_read=True):
        """

        :rtype: boolean -> success
        :type for_read: boolean
        """
        end_loop = False
        while not end_loop:
            while self.__file_path == '':
                print("\nFile path (without file name): ")
                self.__file_path = raw_input()
                if self.__file_path == '':
                    self.__file_path = "."

                if self.__file_path[-1] != "/":
                    self.__file_path += "/"

                if not os.path.exists(self.__file_path):
                    print("No directory exists with this path!")
                    self.__file_path = ''

            print("")
            print("Path : " + self.__file_path)
            print("File name (write 'path' to redefined path, 'exit' to quit): ")
            self.__file_name = raw_input()

            if self.__file_name == 'path':
                self.__file_path = ''
                end_loop = False
            else:
                if self.__file_name != '' and self.__file_name != 'path':
                    end_loop = True

                if self.__file_name != 'path' and self.__file_name != 'exit' and for_read and not self.__exist_file():
                    print("No file exists with this name!")
                    end_loop = False

        return self.__file_name != "exit"

    def __get_file_name(self):
        if not self.__get_file_name_for_read(for_read=False):
            return False

        self.__file_suffix = 1
        while os.path.exists(self.__file_path + self.__file_name + str(self.__file_suffix) + ".png"):
            self.__file_suffix += 1

        return True

    def read_color_img_from_file(self):
        if self.__load_color_from_file == 0:
            name_total_color = self.__file_path + self.__file_name + ".png"
            if not os.path.exists(name_total_color):
                print("Error: no file named " + name_total_color + "\n")
                return np.array([])

            self.__color_img = cv2.imread(name_total_color, cv2.IMREAD_COLOR)

            self.__load_color_from_file = 1

        return self.__color_img

    def read_depth_img_from_file(self):

        if self.__load_depth_from_file == 0:
            name_total_depth = self.__file_path + self.__file_name + "_depthMax"
            depth = 0

            while (not os.path.exists(name_total_depth + str(depth) + ".png")) and depth <= DEPTH_MAX_VALUE:
                # print(nameTotal + str(depth) + ".png") # __DEBUG__
                depth += 1

            if depth > DEPTH_MAX_VALUE or self.__file_name == 'exit':
                print("Error: no file named " + name_total_depth + "*.png\n")
                return

            name_total_depth += str(depth) + ".png"

            img = cv2.imread(name_total_depth, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.__depth_img = img / 255.0 * depth
                self.__depth_img = np.array(self.__depth_img, dtype=np.float32)
                self.__load_depth_from_file = 1

        return self.__depth_img

    def init_read_img_from_file(self, f_path='', f_name=''):
        if f_path == '' or f_name == '':
            self.__get_file_name_for_read(for_read=True)
        elif os.path.exists(f_path+f_name+".png"):
            self.__file_path = f_path
            self.__file_name = f_name
        else:
            self.__file_name = 'exit'

        if self.__file_name == 'exit':
            print("\nNo image loaded!\n")
            return False

        self.__load_color_from_file = 0
        self.__load_depth_from_file = 0

        return True

    def __clean(self, img, n, depth):
        # set the non-finite values (NaN, inf) to n
        # returns 1 where the img is finite and 0 where it is not
        mask = np.isfinite(img)
        #  where mask puts img, else puts n, so where is finite puts img, else puts n
        img = np.where(mask, img, n)
        img = np.where(img < depth, img, n)
        return img

    def __transform_depth_img(self, img, n, depth):
        img = img * n / depth
        img = np.where(img <= n, img, n)
        img.astype(np.uint8)
        return img

    def init_save_images(self):
        self.__get_file_name()

    def save_depth_image_to_file(self, img, depth):
        print("Saving depth image")
        name_total = self.__file_path + self.__file_name + str(self.__file_suffix) + "_depthMax" + str(depth) + ".png"
        img = self.__clean(img, 255, depth)
        img = self.__transform_depth_img(img, 255, depth)
        if cv2.imwrite(name_total, img, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
            print("Depth image saved as : " + name_total + "\n")
            return True
        else:
            print("Error: depth image not saved !")
            return False

    def save_color_image_to_file(self, img):
        print("Saving color image")
        name_total = self.__file_path + self.__file_name + str(self.__file_suffix) + ".png"
        if cv2.imwrite(name_total, img, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
            print("Color image saved as : " + name_total + "\n")
            return True
        else:
            print("Error: color image not saved !\n")
            return False


#############
# Constants #
#############

DEPTH_MAX_VALUE = 10

fileM = FileManager()


#############
# Functions #
#############

def initialisation_file_reading(f_path='', f_name=''):
    return fileM.init_read_img_from_file(f_path, f_name)


def read_color_img():
    return fileM.read_color_img_from_file()


def read_depth_img():
    return fileM.read_depth_img_from_file()


def initialisation_img_saving():
    return fileM.init_save_images()


def save_color_img(img):
    return fileM.save_color_image_to_file(img)


def save_depth_img(img, depth):
    return fileM.save_depth_image_to_file(img, depth)

