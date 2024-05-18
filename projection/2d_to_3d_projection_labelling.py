import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from project_pc_to_img import *


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

    # Iterate through camera images
    for f in img_files:
        # Misalignment issues
        # if f not in ["1651752750486380409.png"]:
        #     continue

        # Overprojection issue
        if f not in ["1651752778120209413.png", "1651752778517883130.png", "1651752787923372972.png"]:
            continue
        
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
        # plt.show()

        # Find the closest img timestamp to the given pc
        hor_diff = np.abs(hor_timestamps - np.repeat(int(f[:-4]), hor_timestamps.shape))
        hor_idx = np.argmin(hor_diff)

        tele_diff = np.abs(tele_timestamps - np.repeat(int(f[:-4]), tele_timestamps.shape))
        tele_idx = np.argmax(tele_diff)

        # Load pc and image
        hor_pc = np.load(os.path.join(hor_dir, hor_files[hor_idx]))
        tele_pc = np.load(os.path.join(tele_dir, tele_files[tele_idx]))
        img = plt.imread(os.path.join(img_dir, f))

        # Print files
        print("LiDAR horizon File:", hor_files[hor_idx])
        print("LiDAR tele_15 File:", tele_files[tele_idx])
        print("Image File:", f)
        print()

        # Get projection of point clouds onto images
        full_hor_pts, hor_pts = project_pc_to_img(hor_pc, img, hor_ext_mat, cam_ext_mat, cam_int_mat)
        full_tele_pts, tele_pts = project_pc_to_img(tele_pc, img, tele_ext_mat, cam_ext_mat, cam_int_mat)

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
        # implot = plt.imshow(img)
        plt.scatter(hor_pts[:, 0], hor_pts[:, 1], marker='.', color='red', alpha=1, s=0.5)
        plt.scatter(tele_pts[:, 0], tele_pts[:, 1], marker='.', color='blue', alpha=1, s=0.5)
        plt.title("img: {}, horizon: {}, tele_15: {}".format(f, hor_files[hor_idx], tele_files[tele_idx]))
        plt.scatter(hor_pole_idx[:, 0], hor_pole_idx[:, 1], marker='.', color='lime', alpha=1, s=0.5)
        plt.scatter(tele_pole_idx[:, 0], tele_pole_idx[:, 1], marker='.', color='cyan', alpha=1, s=0.5)
        plt.show()

        # Find the 3D points corresponding to the point cloud and pole labels
        hor_idx = []
        hor_else_idx = []
        for i in range(full_hor_pts.shape[0]):
            x = full_hor_pts[i][0]
            y = full_hor_pts[i][1]
            for j in range(hor_pole_idx.shape[0]):
                if hor_pole_idx[j][0] == x and hor_pole_idx[j][1] == y:
                    hor_idx.append(i)
                else:
                    hor_else_idx.append(i)
        
        tele_idx = []
        tele_else_idx = []
        for i in range(full_tele_pts.shape[0]):
            x = full_tele_pts[i][0]
            y = full_tele_pts[i][1]
            for j in range(tele_pole_idx.shape[0]):
                if tele_pole_idx[j][0] == x and tele_pole_idx[j][1] == y:
                    tele_idx.append(i)
                else:
                    tele_else_idx.append(i)

        hor_poles_pcd = o3d.geometry.PointCloud()
        hor_poles_pcd.points = o3d.utility.Vector3dVector(hor_pc[hor_idx, :3])
        hor_poles_pcd.paint_uniform_color([1, 0, 0])

        hor_else_pcd = o3d.geometry.PointCloud()
        hor_else = hor_pc[hor_else_idx, :3]
        hor_else = hor_else[hor_else[:, 2] > -1.3]
        hor_else_pcd.points = o3d.utility.Vector3dVector(hor_else)
        hor_else_pcd.paint_uniform_color([0, 1, 0])

        # tele_poles_pcd = o3d.geometry.PointCloud()
        # tele_poles_pcd.points = o3d.utility.Vector3dVector(tele_pc[tele_idx, :3])
        # tele_poles_pcd.paint_uniform_color([1, 1, 0])

        # tele_else_pcd = o3d.geometry.PointCloud()
        # tele_else_pcd.points = o3d.utility.Vector3dVector(tele_pc[tele_else_idx, :3])
        # tele_else_pcd.paint_uniform_color([0, 1, 0])

        print("Poles:", hor_poles_pcd)
        print("Other Points:", hor_else_pcd)
        o3d.visualization.draw_geometries([hor_poles_pcd, hor_else_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])
