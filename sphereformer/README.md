# sphereformer
## `intensity_dist.py`
This file is used for visualizing the distribution of the intensity on the Siemens HVLE and SemanticKITTI datasets.

## `redistribute_intensity.py`
This file normalizes the intensity values of the Siemens HVLE dataset to [0, 1]. It also scales and translates the scene of HVLE to better match with that of KITTI.

## `visualize_sf_kitti.py` & `visualize_sf_nuscenes.py`
These scripts can be used to visualize the LiDAR semantic segmentation inference results of SphereFormer on the SemanticKITTI and nuScenes-trained models, respectively.
