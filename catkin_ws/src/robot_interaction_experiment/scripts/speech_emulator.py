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
import time
import os, pickle, copy
''' This script helps automate the database recording by simulating speech.'''

# publishes the given words describing the image
def send_words(words):
    global repeated_dic
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
    time.sleep(0.4)
    object_pub.publish(audio)


if __name__ == '__main__':
    repeated_dic = list()
    rospy.init_node('speech_emu', anonymous=True)
    typying = False
    audio = Audition_Features()
    object_pub = rospy.Publisher("audition_features", Audition_Features, queue_size=1)
    sentence_structs = list()
    sentence_structs.append('This is a ')
    sentence_structs.append('That is a ')
    sentence_structs.append('Here we have a ')
    sentence_structs.append('I can see a ')
    sentence_structs.append('In front of me there is ')
    done = list()
    colors = ['yellow', 'red', 'purple', 'cyan', 'green', 'blue']
    filename = 'obj_description.pickle'
    os.remove(filename)
    # gets the object description from obj_description.pickle and prompts the user to accept or reject the suggestion
    while 1:
        if typying:
            clr_n_label = (raw_input('Describe the scene (color + obj name): ')).split()
            for sentence in sentence_structs:
                words = sentence
                words = words.split()
                words += clr_n_label
                send_words(words)
                print (words)
        else:
            # keep checking until there is a new file with the description
            filename = 'obj_description.pickle'
            while not os.path.isfile('obj_description.pickle'):
                time.sleep(0.4)
                print ('No file =(')
            with open(filename, 'r') as f:
                sentence_structs = pickle.load(f)

            sentence_cp = copy.copy(sentence_structs)
            if (sentence_cp[0].split()[-2] in done) is True:
                # checks if already recorded this color
                print ('Already have this color: ' + str(sentence_cp[0].split()[-2]) + ', deleting file to get a new one.')
                print ('Missing: ' + str([obj for obj in colors if obj not in done]))
                not_recorded = [obj for obj in colors if obj not in done]
                if len(not_recorded) == 0:
                    print ('All done here, quitting...')
                    exit(1)
                os.remove(filename)
                time.sleep(0.2)
                continue
            print (sentence_cp[0].split()[-2] + ' ' + sentence_cp[0].split()[-1])
            print ('Missing: ' + str([obj for obj in colors if obj not in done]))
            y_n = str(raw_input('Send those? y/n: '))
            if y_n == 'y':
                for sentence in sentence_structs:
                    words = sentence
                    words = words.split()
                    print (words)
                    send_words(words)
                    time.sleep(0.4)
                print ('DONE!')
                time.sleep(3)
                done.append(words[-1])
            else:
                print 'Removed it'
            os.remove(filename)
            while not os.path.isfile('obj_description.pickle'):
                time.sleep(0.4)
