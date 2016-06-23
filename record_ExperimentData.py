#!/usr/bin/env python
import threading
from cv_bridge import CvBridge, CvBridgeError
from robot_interaction_experiment.msg import Audition_Features
import cv2
import rospy
from robot_interaction_experiment.msg import Detected_Objects_List

''' This script is used to record the data generated during the experiments so it can later be used for training.
The recorded data is:
    The detected objects as published by the detected_objects_list topic
    Each object histograms: Shape / HoG / Color
    Each object associated words and the current dictionary as per the topic audition_features '''

lock = threading.Lock()
dictionary = ""
got_speech = 0
words_histogram = []
speech = []
speech_counter = 0
f = open('RecordedData/ExperimentDataLog', 'w', 0)

'''
Callback from the vision related topic. Waits until there is a speech to associate it to the detected object and
record.
'''


def callback_rgb(data):
    global lock
    lock.acquire()  # Prevent data races
    try:
        global got_speech
        global speech_counter
        if got_speech == 0:
            return
        counter = 0
        print 'Recording experiment ' + str(speech_counter) + '\n'
        f.write('#Informations about sample, No. ' + str(speech_counter) + '\n')
        f.write('#Format = Dictionary / Detected Words / Shape / HoG / Color / Words Histogram\n')
        for det_object in data.detected_objects_list:
            f.write('#Object ' + str(det_object.id) + '\n')
            f.write(', '.join(map(str, dictionary)))
            f.write('\n')
            f.write(', '.join(map(str, speech)) + '\n')
            f.write(', '.join(map(str, det_object.features.shape_histogram)) + '\n')
            print len(det_object.features.shape_histogram)
            print len(det_object.features.hog_histogram)
            print len(det_object.features.colors_histogram)

            f.write(', '.join(map(str, det_object.features.hog_histogram)) + '\n')
            f.write(', '.join(map(str, det_object.features.colors_histogram)) + '\n')
            f.write(', '.join(map(str, words_histogram)) + '\n\n')
            try:
                object_image = CvBridge().imgmsg_to_cv2(det_object.image)
            except CvBridgeError, e:
                print e
                return
            cv2.imwrite('RecordedData/Exp_' + str(speech_counter) + '_objs_' + str(det_object.id) + '.png', object_image)
        speech_counter += 1
        got_speech = 0
    finally:
        lock.release()

'''
Callback from the audio related topic. Receives the current dictionary, words histogram and last spoken words (speech).
'''


def callback_audio_recognition(words):
    global dictionary
    global words_histogram
    global speech
    global got_speech
    if not words.complete_words:
        return
    speech = words.complete_words
    got_speech = 1
    dictionary = words.words_dictionary
    words_histogram = words.words_histogram


if __name__ == '__main__':
    rospy.init_node('recordExperimentData', anonymous=True)
    object_sub = rospy.Subscriber("audition_features", Audition_Features, callback_audio_recognition)
    object_sub = rospy.Subscriber("detected_objects_list", Detected_Objects_List, callback_rgb)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
