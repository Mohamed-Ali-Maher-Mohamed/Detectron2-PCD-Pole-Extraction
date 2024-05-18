# pole-extraction
This respository contains work completed by Kelly Zhu during her DAAD RISE 2023 internship. 

It contains code to extract poles from LiDAR point clouds by projecting the point cloud to the image plane and segmenting the projected point cloud using a fused image segmentation technique with PanopticFCN. The output of the pipeline is a set of 3D bounding boxes around the detected poles in the point cloud.

## clustering
This folder contains scripts for performing DBSCAN clustering on the LiDAR point cloud scene.

## detectron2
This folder contains the repository of [detectron2](https://github.com/facebookresearch/detectron2), which was used for panoptic segmentation on images using the [PanopticFCN](https://github.com/dvlab-research/PanopticFCN) network. Minor modifications were made to the codebase so that it could run both a COCO-trained and Cityscapes-trained model.

This respository also contains a new folder titled `fusion`, which contains the code for fusing the segmented traffic features (mainly poles) from the Cityscapes model into the segmentation results of the COCO-trained model.

## data_parsing
This folder contains scripts for parsing and extracting the db3 ROS bag files from the Siemens HVLE dataset.

## projection
This folder contains scripts for projecting the 3D LiDAR point clouds onto the 2D image plane. It contains the code for extracting the extrinsic and intrinsic matrices, projecting the points onto the image, isolating the poles from the point cloud scene, and the final method for pole extraction.

The main contributions of the summer can be found in this folder.

## sphereformer
This folder contains the respository of [SphereFormer](https://github.com/dvlab-research/SphereFormer), which was used for testing LiDAR segmentation on the Siemens HVLE dataset using the pre-trained SphereFormer models. It also contains additional scripts for visualizing segmentation results and visualizing/normalizing intensity distributions.
