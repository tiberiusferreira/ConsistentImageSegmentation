# ConsistentImageSegmentation

Several scripts which take an RGB-D camera feed and microphone sound. 

The goal is to segment the shown cubes and always show the image on the cubes face in the same orientation.

The cubes have different images, but all the faces of one given cube have the same image with different color, as shown below.

![alt tag](https://github.com/tiberiusferreira/ConsistentImageSegmentation/blob/master/catkin_ws/src/robot_interaction_experiment/scripts/rapport/cubes_photo.jpeg?raw=true)

The segmentation is simple just by taking the depth image, resizing/cropping it so it has the same aspect ratio of the RGB one, finding shapes which resemble a cube and mapping it to the RGB one. 

The hard part is finding the correct orientation of the image (by correct I mean always showing the same image shape in the same orientation even when it changes colors or its image is a bit deformed due to camera position. 

Three methods were implemented:

HSV Filtering to find the image contours and using the contour to orientate the object (by finding a minimal area enclosing triangle enclosing the contour points for example)

![alt tag](https://github.com/tiberiusferreira/ConsistentImageSegmentation/blob/master/catkin_ws/src/robot_interaction_experiment/scripts/rapport/DiffTreatments.png?raw=true)

Using Sobel Filtering to find the image contours and Principal Component Analysis (PCA), Sift features and contour points to orientate it.

![alt tag](https://github.com/tiberiusferreira/ConsistentImageSegmentation/blob/master/catkin_ws/src/robot_interaction_experiment/scripts/rapport/PCA_ex.png?raw=true)


Using Machine Learning based on pre-recorded images of the cubes faces. The classifier is a Stochastic Gradient Descent using logistic regression as loss function implemented using the Scikit toolkit http://scikit-learn.org/. This works quite well since the shapes can be very separated well into classes as can be seen below.

![alt tag](https://github.com/tiberiusferreira/ConsistentImageSegmentation/blob/master/catkin_ws/src/robot_interaction_experiment/scripts/rapport/classes.png?raw=true)

The other scripts accomplish small tasks aimed at helping a bigger project: Cross-situational noun and adjective learning in an interactive scenario, 2015 Joint IEEE International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob)


More information can be obtained reading the full report rapport-PRE-ReleaseCandidate2.pdf in the directory catkin_ws/src/robot_interaction_experiment/scripts/rapport
