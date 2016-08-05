# Consistent Image Segmentation

Several scripts which take an RGB-D (color and depth data) camera feed and microphone sound. 

The goal is to segment the shown cubes and always show the image on the cubes face in the same orientation.

The cubes have different images, but all the faces of one given cube have the same image with different color, as shown below.

![alt tag](https://github.com/tiberiusferreira/ConsistentImageSegmentation/blob/master/catkin_ws/src/robot_interaction_experiment/scripts/rapport/cubes_photo.jpeg?raw=true)

Fot the segmentation the depth image is resized and cropped in order to have the same aspect ratio of the RGB one. After that shapes which resemble a cube are found using OpenCV's findContours function. After that the corresponding regions are taking from the RGB image. 

The hard part is finding the correct orientation of the image (correct meaning always showing the same image shape in the same orientation even when it changes colors or its image is a bit deformed due to camera position). 

Three methods were implemented:

HSV Filtering to find the image contours and using the contour to orientate the object (by finding a minimal area enclosing triangle enclosing the contour points for example)

![alt tag](https://github.com/tiberiusferreira/ConsistentImageSegmentation/blob/master/catkin_ws/src/robot_interaction_experiment/scripts/rapport/DiffTreatments.png?raw=true)

Using Sobel Filtering to find the image contours and Principal Component Analysis (PCA), Sift features and contour points to orientate it.

![alt tag](https://github.com/tiberiusferreira/ConsistentImageSegmentation/blob/master/catkin_ws/src/robot_interaction_experiment/scripts/rapport/PCA_ex.png?raw=true)


Using Machine Learning based on pre-recorded images of the cubes faces. The classifier is a Stochastic Gradient Descent using logistic regression as loss function implemented using the Scikit toolkit http://scikit-learn.org/. This works quite well since the shapes can be very separated well into classes as can be seen below.

![alt tag](https://github.com/tiberiusferreira/ConsistentImageSegmentation/blob/master/catkin_ws/src/robot_interaction_experiment/scripts/rapport/classes.png?raw=true)

The other scripts accomplish small tasks aimed at helping a bigger project: Cross-situational noun and adjective learning in an interactive scenario, 2015 Joint IEEE International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob). One of those smaller tasks was to record an object database containing the objects HoG, Color Histogram, words describing, its color image and an image representing the HoG. A visualization tool was also created, a sample of its results is shown below. 

![alt tag](https://github.com/tiberiusferreira/ConsistentImageSegmentation/blob/master/catkin_ws/src/robot_interaction_experiment/scripts/Data_visualization_example.png?raw=true)


More information can be obtained reading the full report rapport-PRE-ReleaseCandidate2.pdf in the directory catkin_ws/src/robot_interaction_experiment/scripts/rapport
