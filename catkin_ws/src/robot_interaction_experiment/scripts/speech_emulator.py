#!/usr/bin/env python
import threading
from cv_bridge import CvBridge, CvBridgeError
from robot_interaction_experiment.msg import Audition_Features
import cv2
import string
import rospy
from robot_interaction_experiment.msg import Detected_Objects_List
import numpy as np
from collections import Counter

if __name__ == '__main__':
    rospy.init_node('speech_emu', anonymous=True)
    repeated_dic = list()
    audio = Audition_Features()
    object_pub = rospy.Publisher("audition_features", Audition_Features, queue_size=1)
    while 1:
        words = (raw_input('Describe the scene: ')).split()
        audio.complete_words = words
        repeated_dic.extend(words)
        _, idx = np.unique(repeated_dic, return_index=True)
        repeated_dic = np.array(repeated_dic)
        audio.words_dictionary = repeated_dic[np.sort(idx)]
        repeated_dic = repeated_dic.tolist()
        dictionary = np.zeros(len(audio.words_dictionary))
        for words in audio.words_dictionary:
            dictionary[audio.words_dictionary.tolist().index(words)] = Counter(repeated_dic)[words]
        audio.words_histogram = dictionary
        object_pub.publish(audio)

