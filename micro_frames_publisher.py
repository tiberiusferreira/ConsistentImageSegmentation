#!/usr/bin/env python

import pyaudio
import rospy
import std_msgs.msg

def frames_publisher():
    pub = rospy.Publisher('micro_frames', std_msgs.msg.String)
    
    chunk = int(rospy.get_param('~chunk', '2048'))
    rate  = int(rospy.get_param('~rate',  '16000'))
    print 'chunk:', chunk
    print 'rate: ', rate
    print
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)
    
    while not rospy.is_shutdown():
        frame = stream.read(chunk)
        pub.publish(frame)

    stream.stop_stream()
    stream.close()
    audio.terminate()

if __name__ == '__main__':
    rospy.init_node('micro_frames_publisher', anonymous=True)
    frames_publisher()
