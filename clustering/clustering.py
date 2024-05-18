import os
import pickle
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from project_pc_to_img import *


def cantor(a, b):
    '''
    Returns a unique hash for any two non-neg ints using the Cantor pairing function
    Adapted to take arrays as input as well (multiple hashings at once)

    Cantor pairing function: (a + b) * (a + b + 1) / 2 + a; where a, b >= 0

    input:
        a: (n, 1) array containing non-negative integers
        b: (n, 1) array containing non-negative integers

    returns:
        hashes: (n, 1) array containing the hashes
    '''
    numerator = np.multiply((a + b), (a + b + 1))
    denominator = 2 + a

    hashes = np.divide(numerator, denominator)
    return hashes


if __name__ == '__main__':
    pc_dir = '/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_horizon'
    pc_files = sorted(os.listdir(pc_dir))
    
    # # LiDAR files associated with the images used in remove_overprojection.py
    pc_files = ['1651752778125636279.npy', '1651752778534021260.npy', '1651752787926256152.npy']

    # # All LiDAR files associated with the segmented images
    # pc_files = ['1651752733268364234.npy', '1651752733470564360.npy', '1651752733631597404.npy', 
    #             '1651752750533615583.npy', '1651752770136747616.npy', '1651752778125636279.npy',
    #             '1651752778534021260.npy', '1651752787926256152.npy']
    
    # Path to segmented images
    seg_label_dir = "/Users/kelly/Documents/hsd/detectron2/output/siemens_inference"

    # Path to datasets
    hor_dir = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_horizon"
    tele_dir = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_tele_15"
    img_dir = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/segmented_cam_imgs"

    hor_files = sorted(os.listdir(hor_dir))
    tele_files = sorted(os.listdir(tele_dir))
    img_files = sorted(os.listdir(img_dir))

    img_timestamps = np.array([int(f[:-4]) for f in img_files])

    # Load bag file
    bag_path = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11"
    rosbags_register_types_from_messages_tar(Path(bag_path) / "attachments/message_definitions.tar")

    # Read bag file for extrinsic and intrinsic matrices
    hor_topic = "/lidar_horizon/lidar/data_cal"
    tele_topic = "/lidar_tele_15/lidar/data_cal"
    hor_ext_mat = get_extrinsic_matrix(hor_topic, bag_path)
    tele_ext_mat = get_extrinsic_matrix(tele_topic, bag_path)

    cam_topic = "/camera_0/data_cal"
    cam_ext_mat = get_extrinsic_matrix(cam_topic, bag_path)
    cam_int_mat = get_intrinsic_matrix(cam_topic, bag_path)

    for f in pc_files:
        # f = "1651752778125636279.npy"
        pc_path = os.path.join(pc_dir, f)
        pc = np.load(pc_path)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

        # o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])
        
        # Extract xyz points
        points = pc[:, :3]

        # Project the points onto the image
        # First, find the closest image timestamp to the LiDAR file
        img_diff = np.abs(img_timestamps - np.repeat(int(f[:-4]), img_timestamps.shape))
        img_idx = np.argmin(img_diff)
        img_file = img_files[img_idx]
        img = plt.imread(os.path.join(img_dir, img_files[img_idx]))

        # Load COCO segmentation results (get vegetation labels)
        coco_pred_path = os.path.join(seg_label_dir, img_file[:-4], img_file[:-4] + "_fused_pan_seg.pkl")

        with open(coco_pred_path, "rb") as file:
            coco_pred = pickle.load(file)

        # Retrieve the panoptic segmentation from Cityscapes and COCO model
        pan_seg = coco_pred["pan_seg"]
        seg_info = coco_pred["seg_info"]

        # Get projection of point clouds on image
        full_proj_pts, proj_pts = project_pc_to_img(points, img, hor_ext_mat, cam_ext_mat, cam_int_mat)
        
        # Get hash table
        hashes = cantor(full_proj_pts[:, 0], full_proj_pts[:, 1])
        indices = np.arange(full_proj_pts.shape[0])
        hash_table = {h:i for i,h in enumerate(hashes)}

        # Iterate through each category and find the one corresponding to vegetation (category_id = 8)
        for info in seg_info:
            if info["category_id"] == 37:
                id = info["id"]
                area = info["area"]
        
        # Obtain a mask of the vegetation
        mask = pan_seg == id
        plt.imshow(mask)
        # plt.show()

        # Using the vegetation mask, assign a label to each point (belongs to vegetation or not)
        proj_pts_img = np.zeros(mask.shape)

        # Assign all points to 1
        proj_pts_img[proj_pts[:, 1], proj_pts[:, 0]] = 1

         # Mask the lidar points with the inv veg mask (logical and)
        proj_pts_img = np.logical_and(proj_pts_img, ~mask)

        # Grab the indices of point clouds within the pole regions
        non_veg_idx = proj_pts_img.nonzero()
        non_veg_idx = np.vstack((non_veg_idx[1], non_veg_idx[0])).T

        # Visualize just the segmented non-veg points in the image
        plt.scatter(proj_pts[:, 0], proj_pts[:, 1], marker='.', color='red', alpha=1, s=0.5)
        plt.xlim(0, proj_pts_img.shape[1])
        plt.ylim(0, proj_pts_img.shape[0])
        plt.gca().invert_yaxis()
        plt.show()

        # Find the 3D points corresponding to the point cloud and pole labels
        pcd_idx = []
        pcd_else_idx = []

        # Use the hash table to recover the indices of the 3d points
        non_veg_hashes = cantor(non_veg_idx[:, 0], non_veg_idx[:, 1])
        for i in non_veg_hashes:
            pcd_idx.append(hash_table[i])
        
        # Get the remaining indices belonging to vegetation
        pcd_else_idx = np.setxor1d(indices, pcd_idx)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[pcd_idx])
        pcd.paint_uniform_color([1, 0, 0])
        
        else_pcd = o3d.geometry.PointCloud()
        else_pcd.points = o3d.utility.Vector3dVector(points[pcd_else_idx])
        else_pcd.paint_uniform_color([0, 0, 1])

        # o3d.visualization.draw_geometries([pcd, else_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])
        
        full_pcd = o3d.geometry.PointCloud()
        full_pcd.points = o3d.utility.Vector3dVector(points)

        points = points[pcd_idx]

        # Remove ground plane points
        mask = points[:, 2] > -1.3
        landmarks = o3d.geometry.PointCloud()
        landmarks.points = o3d.utility.Vector3dVector(points[mask])

        ground = o3d.geometry.PointCloud()
        ground.points = o3d.utility.Vector3dVector(points[~mask])
        ground.paint_uniform_color([1, 0, 0])

        # o3d.visualization.draw_geometries([landmarks, ground, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])

        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                landmarks.cluster_dbscan(eps=4, min_points=10, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        landmarks.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([landmarks, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])

        # break
