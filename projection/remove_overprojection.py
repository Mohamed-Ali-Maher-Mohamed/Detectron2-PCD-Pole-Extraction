import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster import DBSCAN

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


if __name__ == "__main__":
    # Path to segmented images
    seg_label_dir = "/Users/kelly/Documents/hsd/detectron2/output/siemens_inference"

    # Path to datasets
    hor_dir = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_horizon"
    tele_dir = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_tele_15"
    img_dir = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/segmented_cam_imgs"

    hor_files = sorted(os.listdir(hor_dir))
    tele_files = sorted(os.listdir(tele_dir))
    img_files = sorted(os.listdir(img_dir))

    hor_timestamps = np.array([int(f[:-4]) for f in hor_files])
    tele_timestamps = np.array([int(f[:-4]) for f in tele_files])

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

    print(img_files)

    # Iterate through camera images
    for f in img_files:
        # Misalignment issues
        # if f not in ["1651752750486380409.png"]:
        #     continue

        # Overprojection issue
        # if f not in ["1651752778120209413.png", "1651752778517883130.png", "1651752787923372972.png"]:
        #     continue

        # Small issue with extracting pole layer in 1651752770131145643.png
        # if f not in ["1651752770131145643.png"]:
        #     continue
        
        # Load PanopticFCN_cityscapes segmentation results (get pole labels)
        cityscapes_pred_path = os.path.join(seg_label_dir, f[:-4], f[:-4] + "_cityscapes_pan_seg.pkl")

        with open(cityscapes_pred_path, "rb") as file:
            cityscapes_pred = pickle.load(file)

         # Retrieve the panoptic segmentation from Cityscapes and COCO model
        pan_seg = cityscapes_pred["pan_seg"]
        seg_info = cityscapes_pred["seg_info"]

        # Iterate through each category and find the one corresponding to poles (category_id = 5)
        for info in seg_info:
            if info["category_id"] == 5:
                id = info["id"]
                area = info["area"]
        
        # Obtain a mask of the poles
        mask = pan_seg == id
        plt.imshow(mask)
        plt.show()

        # Find the closest img timestamp to the given pc
        hor_diff = np.abs(hor_timestamps - np.repeat(int(f[:-4]), hor_timestamps.shape))
        hor_idx = np.argmin(hor_diff)

        tele_diff = np.abs(tele_timestamps - np.repeat(int(f[:-4]), tele_timestamps.shape))
        tele_idx = np.argmax(tele_diff)

        # Load pc and image
        hor_pc = np.load(os.path.join(hor_dir, hor_files[hor_idx]))
        tele_pc = np.load(os.path.join(tele_dir, tele_files[tele_idx]))
        img = plt.imread(os.path.join(img_dir, f))

        hor_pcd = o3d.geometry.PointCloud()
        hor_pcd.points = o3d.utility.Vector3dVector(hor_pc[:, :3])

        # Print files
        print("LiDAR horizon File:", hor_files[hor_idx])
        print("LiDAR tele_15 File:", tele_files[tele_idx])
        print("Image File:", f)
        print()

        # Get projection of point clouds onto images
        full_hor_pts, hor_pts = project_pc_to_img(hor_pc, img, hor_ext_mat, cam_ext_mat, cam_int_mat)
        full_tele_pts, tele_pts = project_pc_to_img(tele_pc, img, tele_ext_mat, cam_ext_mat, cam_int_mat)

        # Get hash table
        hashes = cantor(full_hor_pts[:, 0], full_hor_pts[:, 1])
        indices = np.arange(full_hor_pts.shape[0])
        hash_table = {h:i for i,h in enumerate(hashes)}

        # Using the pole mask, assign a label to each point (belongs to pole or doesn't belong to pole)
        hor_img = np.zeros(mask.shape)
        tele_img = np.zeros(mask.shape)

        # Assign all points to 1
        hor_img[hor_pts[:, 1], hor_pts[:, 0]] = 1
        tele_img[tele_pts[:, 1], tele_pts[:, 0]] = 1

        # Mask the lidar points with the pole mask (logical and)
        hor_img = np.logical_and(hor_img, mask)
        tele_img = np.logical_and(tele_img, mask)

        # Grab the indices of point clouds within the pole regions
        hor_pole_idx = hor_img.nonzero()
        tele_pole_idx = tele_img.nonzero()

        hor_pole_idx = np.vstack((hor_pole_idx[1], hor_pole_idx[0])).T
        tele_pole_idx = np.vstack((tele_pole_idx[1], tele_pole_idx[0])).T

        # Visualize projection on image
        # # implot = plt.imshow(img)
        # plt.scatter(hor_pts[:, 0], hor_pts[:, 1], marker='.', color='lime', alpha=1, s=0.5)
        # # plt.scatter(tele_pts[:, 0], tele_pts[:, 1], marker='.', color='blue', alpha=1, s=0.5)
        # plt.title("img: {}, horizon: {}, tele_15: {}".format(f, hor_files[hor_idx], tele_files[tele_idx]))
        # plt.scatter(hor_pole_idx[:, 0], hor_pole_idx[:, 1], marker='.', color='red', alpha=1, s=0.5)
        # # plt.scatter(tele_pole_idx[:, 0], tele_pole_idx[:, 1], marker='.', color='cyan', alpha=1, s=0.5)
        # plt.show()

        # Visualize just the segmented pole points in the image
        # plt.scatter(hor_pole_idx[:, 0], hor_pole_idx[:, 1], marker='.', color='red', alpha=1, s=0.5)
        # plt.title("img: {}, horizon: {}, tele_15: {}".format(f, hor_files[hor_idx], tele_files[tele_idx]))
        # plt.xlim(0, hor_img.shape[1])
        # plt.ylim(0, hor_img.shape[0])
        # plt.gca().invert_yaxis()
        # plt.show()

        # ---------- 2D IMAGE CLUSTERING HERE ----------
        db = DBSCAN(eps=150, min_samples=5).fit(hor_pole_idx) # 175, 10 is good
        labels = db.labels_
        max_label = labels.max()
        print(f"2d point cloud projection has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0

        plt.scatter(hor_pole_idx[:, 0], hor_pole_idx[:, 1], marker='.', color=colors, alpha=1, s=10)
        plt.title("img: {}, horizon: {}, tele_15: {}".format(f, hor_files[hor_idx], tele_files[tele_idx]))
        plt.xlim(0, hor_img.shape[1])
        plt.ylim(0, hor_img.shape[0])
        plt.gca().invert_yaxis()
        plt.show()

        # Find the 3D points corresponding to the point cloud and pole labels
        hor_idx = []
        hor_else_idx = []

        # Use the hash table to recover the indices of the 3d points
        pole_hashes = cantor(hor_pole_idx[:, 0], hor_pole_idx[:, 1])
        for i in pole_hashes:
            hor_idx.append(hash_table[i])
        
        # Get the remaining indices belonging to non-poles
        hor_else_idx = np.setxor1d(indices, hor_idx)
        
        hor_poles_pcd = o3d.geometry.PointCloud()
        hor_poles = hor_pc[hor_idx, :3]
        hor_poles_pcd.points = o3d.utility.Vector3dVector(hor_poles)
        hor_poles_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        # hor_poles_pcd.paint_uniform_color([1, 0, 0])

        hor_else_pcd = o3d.geometry.PointCloud()
        hor_else = hor_pc[hor_else_idx, :3]
        hor_else = hor_else[hor_else[:, 2] > -1.3] # Filter out ground plane to reduce computational load
        hor_else_pcd.points = o3d.utility.Vector3dVector(hor_else)
        hor_else_pcd.paint_uniform_color([0, 1, 0])

        print("Poles:", hor_poles_pcd)
        print("Other Points:", hor_else_pcd)
        o3d.visualization.draw_geometries([hor_poles_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])

        # ---------- EXTRACT POLE LAYER HERE ----------
        final_poles_pts = []
        final_poles_colors = []
        pole_bboxes = []

        # Iterate through each cluster (each cluster will correspond to ONE pole after processing)
        for i in range(max(labels) + 1):
            # Visualize the cluster
            cluster = hor_poles[labels == i]
            cluster_colors = colors[labels == i, :3]
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
            cluster_pcd.colors = o3d.utility.Vector3dVector(cluster_colors)

            # ***** Visualize the cluster *****
            # o3d.visualization.draw_geometries([cluster_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])

            # Cluster the cluster (we want to extract the layer with only the poles)
            with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
                subcluster_labels = np.array(
                    cluster_pcd.cluster_dbscan(eps=5, min_points=2, print_progress=True))

            max_label = subcluster_labels.max()
            print(f"point cloud has {max_label + 1} clusters")
            subcluster_colors = plt.get_cmap("tab20")(subcluster_labels / (max_label if max_label > 0 else 1))
            subcluster_colors[subcluster_labels < 0] = 0
            cluster_pcd.colors = o3d.utility.Vector3dVector(subcluster_colors[:, :3])

            # ***** Visualize the cluster, colorized by subcluster *****
            # o3d.visualization.draw_geometries([cluster_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])

            # Find the cluster with the min depth (x-axis is depth) AND at least thresh_pts num of points
            # cluster contains all of the layers, cluster_labels contains the labels to segment them
            thresh_pts = 6
            depth_dict = {} # {avg_depth: cluster_label}
            for j in range(max(subcluster_labels) + 1):
                subcluster = cluster[subcluster_labels == j]
                avg_depth = np.mean(subcluster, axis=0)[0]
                depth_dict[avg_depth] = j

            # Iterate through each subcluster, from lowest to highest depth, and extract the layer containing the poles
            for depth in sorted(depth_dict.keys()):
                # Get the cluster corresponding to the depth
                subcluster_label = depth_dict[depth]
                subcluster = cluster[subcluster_labels == subcluster_label]
                subcluster_color = cluster_colors[subcluster_labels == subcluster_label]

                # Check if the subcluster has greater than thresh_pts
                if subcluster.shape[0] > thresh_pts:
                    final_poles_pts.append(subcluster)
                    final_poles_colors.append(subcluster_color)
                    
                    # Create bounding box for the pole
                    bbox = o3d.geometry.AxisAlignedBoundingBox()
                    bbox = bbox.create_from_points(o3d.utility.Vector3dVector(subcluster))
                    pole_bboxes.append(bbox)

                    break
        
        # Arrays containing the points of the poles and their colours (by cluster)
        final_poles_pts = np.vstack(final_poles_pts)
        final_poles_colors = np.vstack(final_poles_colors)

        # Define the o3d point cloud 
        final_poles_pcd = o3d.geometry.PointCloud()
        final_poles_pcd.points = o3d.utility.Vector3dVector(final_poles_pts)
        final_poles_pcd.colors = o3d.utility.Vector3dVector(final_poles_colors)

        # Visualize final pole point clouds
        # o3d.visualization.draw_geometries([final_poles_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])

        # Visualize pole bounding boxes on full point cloud scene
        vis = [hor_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)]
        vis.extend(pole_bboxes)
        o3d.visualization.draw_geometries(vis)

        print(0)

        # break
