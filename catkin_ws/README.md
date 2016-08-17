# Launching the segmentation program
To launch the program execute the ConsistentImageSegmentation/catkin_ws/src/robot_interaction_experiment/scripts/just_camera.sh script.

Alternatively a rosbag file can be used, so camera or additional hardware is required. In this case it suffices to launch the rosbag.sh shell script. It should launch roscore and start playing the rosbag in loop.  

Then launch the imgseg.py script.

The following window will appear:
![alt tag](https://github.com/tiberiusferreira/ConsistentImageSegmentation/blob/master/catkin_ws/src/robot_interaction_experiment/scripts/rapport/example_imgseg.png?raw=true)

Select the working area from the window shown by clicking and drawing the mouse.

If all went well the segmented cube should be shown.

![alt tag](https://github.com/tiberiusferreira/ConsistentImageSegmentation/blob/master/catkin_ws/src/robot_interaction_experiment/scripts/rapport/select_cube_example.png?raw=true)

# Adding cubes to the database

All the cubes are stored in the LRN_IMGS folder. To add a new one make sure the imgseg.py script has it segmented, in other words, make sure a window showing only the image of the new object is shown like the images in the LRN_IMGS folder. In the image above the window in question is the "Img 0" one.

After that click on Save Learn so the image of the new object is saved in the database. The terminal window will ask for a Label and a Color for the object. 

Delete the current classifier in the Classifier folder and the HoG in the HOG_N_LABELS folder. New ones will be generated on the next launch of the imgseg.py. 

# Recording data for Yuxin

To record data for Yuxin (which is stored in the RecordedData folder) launch the imgseg.py script, make sure the desired object is selected, launch the record_ExperimentData.py script. Indicate a folder name where to store the data. 

Either use:

The microphone to record phrases describing the object launching the following scripts in this order: ./micro_frames_publisher.py, ./speech_detector.py, ./speech_recognizer.py. A better is to use the speech_emulator.py script instead of using the microphone.

Or 

Use the speech_emulator.py script. It relies on the machine learning implementation inside the imgseg.py script to detect the object label and color. So the cube needs to have already been recorded in the database. 
