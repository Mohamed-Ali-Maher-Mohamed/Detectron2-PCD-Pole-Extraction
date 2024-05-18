# clustering
This folder contains the primary code for performing clustering using the DBSCAN algorithm on LiDAR point cloud scenes.

## `clustering.py`
This is the main Python script for performing clustering using DBSCAN. As input, it requires the raw LiDAR point clouds and images that were extracted from the recorded Lidar bag file using the script `data_parsing/read_db3.py`. Additionally, it requires the corresponding segmented image output from the COCO-trained PanopticFCN model for each image in the dataset. These segmented images are used for removing the vegetation points from the point clouds. Finally, it requires the path to the original HVLE bag file for loading the extrinsic and intrinsic matrices.

This script iterates through each point cloud file in the dataset and associates it to the nearest image file in the image dataset. Then, the point cloud will be projected to the image plane in order to remove the vegetation points from the point cloud using the segmentation results from the COCO-trained PanopticFCN model. A hash table that uses the Cantor function is implemented to increase the efficiency of this process. To continue, the ground plane points of the LiDAR scene are removed using a simple masking function. Finally, the remaining point cloud is passed into DBSCAN for clustering.

A visualization of the output of `clustering.py` is shown below:  
![Clustering Output](https://github.com/Mohamed-Ali-Maher-Mohamed/Detectron2-PCD-Pole-Extraction/blob/master/Images/extrcted%20poles.png) "Clustering Output"


## `project_pc_to_img.py`
This file contains the functions used for extracting the intrinsic and extrinsic matrices and performing the 3D point cloud to 2D image plane transformation. The original image overlaid with the projected point cloud are shown below.

![3d to 2d projection](https://github.com/Mohamed-Ali-Maher-Mohamed/Detectron2-PCD-Pole-Extraction/blob/master/Images/projectd%203d%20to%202d.png) "3d to 2d projection"


