#!/bin/bash
bash -c "roscore &";
cd /mnt/hgfs/PRE/ConsistentImageSegmentation/catkin_ws; rosbag play --loop -u 40 test_data_2016-07-11-15-10-55.bag

