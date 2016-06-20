#!/usr/bin/env python

import pylab
import rospy
import std_msgs.msg

speech_frames = []
cpt_silences = 0

def speech_detector(msg):
    global speech_frames_pub
    global speech_frames
    global cpt_silences
    global std_threshold
    global nb_silences_max
    
    nb_speech_frames_min = 0 #1
    nb_frames_min        = nb_speech_frames_min + nb_silences_max
    
    frame    = msg.data
    frame_np = pylab.fromstring(frame, 'Int16')
    
    std = pylab.std(frame_np)
    nb_frames = len(speech_frames)
    if std > std_threshold:
        print 'speech?'
        speech_frames.append(frame)
        cpt_silences = nb_silences_max
        if nb_frames == 0:
            status_pub.publish('record')
    else:
        if cpt_silences == 0:
            if nb_frames > nb_frames_min:
                print 'speech detected in', nb_frames - nb_silences_max, 'chunks'
                speech_buffer = ''.join(speech_frames)
                speech_frames_pub.publish(speech_buffer)
                status_pub.publish('stop')
            elif nb_frames != 0:
                print 'nope...'
                status_pub.publish('cancel')
            speech_frames = []
        else:                
            print 'remaining silences:', cpt_silences
            speech_frames.append(frame)
            cpt_silences -= 1

if __name__ == '__main__':
    speech_frames_pub = rospy.Publisher('speech_frames', std_msgs.msg.String, queue_size=10)
    status_pub = rospy.Publisher('speech_status', std_msgs.msg.String)
    rospy.init_node('speech_detector', anonymous=True)
    rospy.Subscriber('micro_frames', std_msgs.msg.String, speech_detector)
    std_threshold = int(rospy.get_param('~threshold', '500'))
    nb_silences_max = int(rospy.get_param('~silences', '4'))
    print 'threshold:', std_threshold
    print 'silences: ', nb_silences_max
    print
    
    while not rospy.is_shutdown():
        figure = pylab.figure(1)
        speech_frames_copy = list(speech_frames) # prevent from modification
        if speech_frames_copy != []:
            figure.clf()
            data = pylab.fromstring(''.join(speech_frames_copy), 'Int16')
            pylab.plot(data)
            pylab.ylim([-20000, 20000])
            pylab.xlim([0, len(data)])
        figure.show()
        pylab.pause(0.000001)

