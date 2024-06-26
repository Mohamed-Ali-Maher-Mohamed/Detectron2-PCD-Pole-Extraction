# projection
## `2d_to_3d_projection_labelling.py`
This file contains old code for associating 2D projection points back to their original point in the 3D point cloud. It is out-of-date now and has been replaced with an updated method that uses a hash table for increased efficiency and speed. 

## `extract_poles.py`
This file contains the main contributions of the summer. It takes in LiDAR point clouds and camera images as input and generates bounding boxes around the detected poles as the output. This is accomplished by a fusion of two pole detection methods. In the first method, the 3D LiDAR points are projected onto the 2D image plane and the segmented image, generated by PanopticFCN_cityscapes, is used to isolate the points belonging to the poles. These points are then associated back their original point in 3D space to generate pole detections.

In the second method, we use a similar projection method to remove the vegetation from the point cloud scene and use a masking technique to further remove the ground plane points. Using this pruned scene, we perform DBSCAN to obtain clusters of the scene. Some of these clusters contain the isolated points belonging to the poles. We use the pole detections in the first method described above to determine which of these clusters are pole detections.

Finally, a rule-based selection algorithm is used to select the optimal bounding box between the two above listed methods for each pole detection. The output of the pipeline, with different coloured bounding boxes indicating which method was used for detection, is shown below:

<img width="482" alt="Screenshot 2023-08-15 at 4 26 24 PM" src="https://github.com/Mohamed-Ali-Maher-Mohamed/Detectron2-PCD-Pole-Extraction/blob/master/Images/extract_poles.png">

## `project_pc_to_img.py`
This file contains functions used by other scripts which are necessary for projecting the point cloud to the image plane.

<img width="600" src="https://github.com/Mohamed-Ali-Maher-Mohamed/Detectron2-PCD-Pole-Extraction/blob/master/Images/project_pcd_to_image.png">
<img width="600" src="https://github.com/Mohamed-Ali-Maher-Mohamed/Detectron2-PCD-Pole-Extraction/blob/master/Images/3d_pcd_onto_2d.png">

## `remove_overprojection.py`
This file contains the primary code for resolving the issue of overprojection, where multiple layers of the point cloud scene are all classified as poles due to error in the image segmentation. It uses an iterative DBSCAN clustering approach to first cluster the projected pole points into individual pole detections, and then a second clustering on each individual pole detection to select the correct layer to keep. The first clustering is done in the projected 2D space while the second clustering is done in 3D space.

A visualization of the clustering results in 2D space and the final results after two rounds of clustering is shown below.

<img width="600" src="https://github.com/Mohamed-Ali-Maher-Mohamed/Detectron2-PCD-Pole-Extraction/blob/master/Images/clear_poles_2d.png">

<img width="821" alt="Screenshot 2023-07-28 at 9 29 12 AM" src="https://github.com/Mohamed-Ali-Maher-Mohamed/Detectron2-PCD-Pole-Extraction/blob/master/Images/clear_poles_3d.png">



