#!/bin/bash
./clean.sh ; gnome-terminal --tab -e "bash -c 'roslaunch publishRGB-Ddata.launch '" --tab -e "bash -c 'rosrun dynamic_reconfigure dynparam set /camera/driver color_mode 1'" --tab -e "bash -c 'rosrun dynamic_reconfigure dynparam set /camera/driver color_depth_synchronization 1'"
