#!/bin/bash
bash -c "roscore &";
cd /mnt/hgfs/PRE/ConsistentImageSegmentation/catkin_ws; rosbag play --loop -u 40 camera_house_2016-08-05-08-39-41.bag

