# Color-Image-Segmentation-OpenCV
This particular project consists of three different implementations of object detection and color segmentation.

The purpose of this exercise is to use the OpenCV libraries in order to locate objects in images and to distinguish different areas of them based on color. The exercise consists of 3 implementations for each of which a separate file has been created.

The first approach concerns the matching of patterns (template matching) under some noise. First we had to take an object that appears many times in an image we had to add noise repeatedly to visualize and evaluate the results. Then we had to add a Gauss filter and compare the results again. The images used for this purpose are the included piece.jpg and pieces.jpg .

For the second implementation (object detection) the same procedure was followed for the images but the purpose of the query was to locate the object using a color histogram with different threshold values. The images used for this purpose are the included present.jpg and presents.jpg .

In the third and final implementation, an image had to be segmented and evaluated based on its color. Initially, an image was selected that captures a situation with the most consistent color in the background. Then the same image through Photoshop was converted to binary annotated and the remaining steps were performed. The images used for this purpose are the included aircraft.jpg and aircraft.jpg .
