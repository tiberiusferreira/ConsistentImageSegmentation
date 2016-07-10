#!/bin/bash
./clean.sh ; gnome-terminal --tab -e "bash -c 'cd /home/tiberio/catkin_ws/src/robot_interaction_experiment/scripts ; ./just_camera.sh '" --tab -e "bash -c 'cd /home/tiberio/catkin_ws/src/robot_interaction_experiment/scripts ; python micro_frames_publisher.py '" --tab -e "bash -c 'cd /home/tiberio/catkin_ws/src/robot_interaction_experiment/scripts ; python speech_detector.py
 ;'" --tab -e "bash -c 'cd /home/tiberio/catkin_ws/src/robot_interaction_experiment/scripts ; python speech_recognizer.py ';"
