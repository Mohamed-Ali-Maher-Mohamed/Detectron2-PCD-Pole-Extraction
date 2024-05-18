import os
import pickle
import numpy as np

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


def remove_vegetation_and_ground_plane(pan_seg, seg_info, proj_pts, hash_table, pc):
    # Iterate through each category and find the one corresponding to vegetation (category_id = 37)
    for info in seg_info:
        if info["category_id"] == 37:
            veg_id = info["id"]
    
    # Obtain a mask of the vegetation
    veg_mask = pan_seg == veg_id
    # plt.imshow(veg_mask)
    # plt.show()

    # Using the vegetation mask, assign a label to each point (belongs to vegetation or not)
    non_veg_img = np.zeros(veg_mask.shape)
    non_veg_img[proj_pts[:, 1], proj_pts[:, 0]] = 1
    non_veg_img = np.logical_and(non_veg_img, ~veg_mask)

    # Grab the indices of point clouds within the non-veg regions
    non_veg_idx = non_veg_img.nonzero()
    non_veg_idx = np.vstack((non_veg_idx[1], non_veg_idx[0])).T

    # Find the 3D points corresponding to the point cloud and pole labels
    non_veg_pcd_idx = []
    veg_pcd_idx = []

    # Use the hash table to recover the indices of the 3d points
    non_veg_hashes = cantor(non_veg_idx[:, 0], non_veg_idx[:, 1])
    for i in non_veg_hashes:
        non_veg_pcd_idx.append(hash_table[i])
    
    # Get the remaining indices belonging to vegetation
    veg_pcd_idx = np.setxor1d(np.arange(pc.shape[0]), non_veg_pcd_idx)

    # Keep only the points that do not belong to vegetation
    points = pc[:, :3]
    points = points[non_veg_pcd_idx]

    # Mask to remove the ground plane
    gp_mask = points[:, 2] > -1.2
    points = points[gp_mask]

    return points


def extract_poles(pan_seg, seg_info, proj_pts, hash_table, pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    
    # Iterate through each category and find the one corresponding to poles (category_id = 5)
    for info in seg_info:
        if info["category_id"] == 5:
            pole_id = info["id"]
    
    # Obtain a mask of the poles
    pole_mask = pan_seg == pole_id
    # plt.imshow(pole_mask)
    # plt.show()

    # Using the pole mask, assign a label to each point (belongs to pole or doesn't belong to pole)
    pole_img = np.zeros(pole_mask.shape)
    pole_img[proj_pts[:, 1], proj_pts[:, 0]] = 1
    pole_img = np.logical_and(pole_img, pole_mask)

    # Grab the indices of point clouds within the pole regions
    pole_idx = pole_img.nonzero()
    pole_idx = np.vstack((pole_idx[1], pole_idx[0])).T

     # ---------- 2D IMAGE CLUSTERING HERE ----------
    db = DBSCAN(eps=150, min_samples=5).fit(pole_idx) # 175, 10 is good
    labels = db.labels_
    max_label = labels.max()
    print(f"2d point cloud projection has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0

    # Find the 3D points corresponding to the point cloud and pole labels
    pole_pcd_idx = []
    non_pole_pcd_idx = []

    # Use the hash table to recover the indices of the 3d points
    pole_hashes = cantor(pole_idx[:, 0], pole_idx[:, 1])
    for i in pole_hashes:
        pole_pcd_idx.append(hash_table[i])
    
    # Get the remaining indices belonging to non-poles
    non_pole_pcd_idx = np.setxor1d(np.arange(pc.shape[0]), pole_idx)

    # Get the most recent pole prediction
    poles = pc[pole_pcd_idx, :3]

     # ---------- EXTRACT POLE LAYER HERE ----------
    final_poles_pts = []
    final_poles_colors = []
    poles_bboxes = []

    # Iterate through each cluster (each cluster will correspond to ONE pole after processing)
    for i in range(max(labels) + 1):
        # Visualize the cluster
        cluster = poles[labels == i]
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
                cluster_pcd.cluster_dbscan(eps=5, min_points=2, print_progress=False))

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
                # final_poles_pts.append(subcluster)
                final_poles_colors.append(subcluster_color)
                
                # Create bounding box for the pole
                bbox = o3d.geometry.AxisAlignedBoundingBox()
                bbox = bbox.create_from_points(o3d.utility.Vector3dVector(subcluster))
                poles_bboxes.append(bbox)

                bbox_pts_idx = bbox.get_point_indices_within_bounding_box(pcd.points)
                bbox_pcd = pcd.select_by_index(bbox_pts_idx)
                final_poles_pts.append(np.array(bbox_pcd.points))

                break
    
    return final_poles_pts, final_poles_colors, poles_bboxes


if __name__ == "__main__":
    # Path to segmented images
    seg_label_dir = "/Users/kelly/Documents/hsd/detectron2/output/siemens_inference"

    # Path to datasets
    pc_dir = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_horizon"
    img_dir = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/segmented_cam_imgs"

    pc_files = sorted(os.listdir(pc_dir))
    img_files = sorted(os.listdir(img_dir))

    pc_timestamps = np.array([int(f[:-4]) for f in pc_files])

    # Load bag file
    bag_path = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11"
    rosbags_register_types_from_messages_tar(Path(bag_path) / "attachments/message_definitions.tar")

    # Read bag file for extrinsic and intrinsic matrices
    lidar_topic = "/lidar_horizon/lidar/data_cal"
    lidar_ext_mat = get_extrinsic_matrix(lidar_topic, bag_path)

    cam_topic = "/camera_0/data_cal"
    cam_ext_mat = get_extrinsic_matrix(cam_topic, bag_path)
    cam_int_mat = get_intrinsic_matrix(cam_topic, bag_path)

    # Iterate through camera images
    for f in img_files:
        # files without misalignment issues
        if f not in ["1651752770131145643.png", "1651752778120209413.png", "1651752778517883130.png", "1651752787923372972.png"]:
            continue

        # Import PanopticFCN Cityscapes and COCO predictions (image segmentation)
        cityscapes_pred_path = os.path.join(seg_label_dir, f[:-4], f[:-4] + "_cityscapes_pan_seg.pkl")
        fused_pred_path = os.path.join(seg_label_dir, f[:-4], f[:-4] + "_fused_pan_seg.pkl")

        with open(cityscapes_pred_path, "rb") as file:
            cityscapes_pred = pickle.load(file)
            cityscapes_pan_seg = cityscapes_pred["pan_seg"]
            cityscapes_seg_info = cityscapes_pred["seg_info"]

        with open(fused_pred_path, "rb") as file:
            fused_pred = pickle.load(file)
            fused_pan_seg = fused_pred["pan_seg"]
            fused_seg_info = fused_pred["seg_info"]

        # Find the closest img timestamp to the given pc
        pc_diff = np.abs(pc_timestamps - np.repeat(int(f[:-4]), pc_timestamps.shape))
        pc_idx = np.argmin(pc_diff)

        # Load pc and images
        pc = np.load(os.path.join(pc_dir, pc_files[pc_idx]))
        img = plt.imread(os.path.join(img_dir, f))
        plt.imshow(img)
        # plt.title("img: {}, pc: {}".format(f, pc_files[pc_idx]))
        # plt.show()

        full_pcd = o3d.geometry.PointCloud()
        full_pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

        print("LiDAR horizon File:", pc_files[pc_idx])
        print("Image File:", f)
        print()

        # Get projection of point clouds onto images
        full_proj_pts, proj_pts = project_pc_to_img(pc, img, lidar_ext_mat, cam_ext_mat, cam_int_mat)

        # Get hash table for backprojection (2D to 3D)
        hashes = cantor(full_proj_pts[:, 0], full_proj_pts[:, 1])
        indices = np.arange(full_proj_pts.shape[0])
        hash_table = {h:i for i,h in enumerate(hashes)}

        # ---------- Get pruned scene without vegetation and ground plane (clustering.py) ----------
        pruned_pts = remove_vegetation_and_ground_plane(fused_pan_seg, fused_seg_info, proj_pts, hash_table, pc)

        pruned_pcd = o3d.geometry.PointCloud()
        pruned_pcd.points = o3d.utility.Vector3dVector(pruned_pts)
        pruned_pcd.paint_uniform_color([0, 0, 0])
        # o3d.visualization.draw_geometries([pruned_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])
        
        # ---------- Get pole segmentation results from overprojection fix method (remove_overprojection.py) ----------
        poles_pts_list, poles_colors_list, pfcn_bboxes = extract_poles(cityscapes_pan_seg, cityscapes_seg_info, proj_pts, hash_table, pc)

        # Arrays containing the points of the poles and their colours (by cluster)
        poles_pts = np.vstack(poles_pts_list)
        poles_colors = np.vstack(poles_colors_list)

        poles_pcd = o3d.geometry.PointCloud()
        poles_pcd.points = o3d.utility.Vector3dVector(np.vstack(poles_pts))
        poles_pcd.colors = o3d.utility.Vector3dVector(np.vstack(poles_colors))
        # o3d.visualization.draw_geometries([poles_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])

        # ---------- Combine pole_points and points ----------
        # o3d.visualization.draw_geometries([full_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])
        # o3d.visualization.draw_geometries([pruned_pcd, poles_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])

        # Create sets for both pruned and poles
        pruned_set = set((tuple(i) for i in pruned_pts))
        poles_set = set((tuple(i) for i in poles_pts))

        fused_set = pruned_set.union(poles_set)
        fused_pts = np.array(list(fused_set))

        # ---------- DBSCAN clustering on combined point cloud ----------
        fused_pcd = o3d.geometry.PointCloud()
        fused_pcd.points = o3d.utility.Vector3dVector(fused_set)
        
        with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
                labels = np.array(
                    fused_pcd.cluster_dbscan(eps=3.5, min_points=10, print_progress=False))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        fused_pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # o3d.visualization.draw_geometries([fused_pcd])

        # ---------- Isolate the cluster of points in associated with poles ----------
        post_processed_bboxes = []
        clustered_bboxes = []
        # Iterate through each pole cluster 
        for pole_pts in poles_pts_list:
            # We want to find the corresponding labels of each pt in pole_pts in fused_pcd
            # Grab the indices of pole_pts in fused_pcd
            idx = np.nonzero(np.in1d(fused_pts.view(dtype='float, float, float').reshape(-1), pole_pts.view(dtype='float, float, float').reshape(-1)))[0]

            # Get the labels of all the points in the poles
            pole_labels = labels[idx]
            
            # Get the most frquently occurring label
            values, counts = np.unique(pole_labels, return_counts=True)
            label = values[np.argmax(counts)]

            if label == -1:
                clustered_bboxes.append(None)
                continue

            # Get all the points in fused_pts with label 
            points = o3d.utility.Vector3dVector(fused_pts[labels == label])
            bbox = o3d.geometry.AxisAlignedBoundingBox()
            bbox = bbox.create_from_points(points)
            bbox.color = np.array([1, 0, 0])

            # Add the bounding boxes to the list
            post_processed_bboxes.append(bbox)
            clustered_bboxes.append(bbox)
        
        # Visualize the projection method bounding boxes along with the post-processed clustering bounding boxes 
        vis = post_processed_bboxes.copy()
        vis.extend(pfcn_bboxes)
        vis.extend([full_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])
        # o3d.visualization.draw_geometries(vis)

        final_bboxes = []

        # ---------- Iterate through the bounding boxes and select ideal one ----------
        for i in range(len(pfcn_bboxes)):
            pfcn_bbox = pfcn_bboxes[i]
            clustered_bbox = clustered_bboxes[i]

            pfcn_extent = pfcn_bbox.get_extent()

            # If the most occuring label after post-processing clustering was -1, check if pfcn_bbox is valid
            if not clustered_bbox:
                # If the bbox predicted from PanopticFCN has a height of at least 5m, accept the prediction
                if pfcn_extent[2] > 5:
                    final_bboxes.append(pfcn_bbox)
                continue
            # Otherwise, we have two valid bounding boxes for this detection and need to select the optimal one
            else:
                clustered_extent = clustered_bbox.get_extent()

                pfcn_valid = False
                clustered_valid = False

                # Check that the height and the width of the bounding box are within a reasonable range
                if (pfcn_extent[1] < 7 and pfcn_extent[1] > 5) and (pfcn_extent[2] < 10 and pfcn_extent[2] > 7):
                    pfcn_valid = True
                if (clustered_extent[1] < 7 and clustered_extent[1] > 5) and (clustered_extent[2] < 10 and clustered_extent[2] > 7):
                    clustered_valid = True
                
                # If only one point bounding box is valid, simply select the valid one
                if pfcn_valid and not clustered_valid:
                    final_bboxes.append(pfcn_bbox)
                elif clustered_valid and not pfcn_valid:
                    final_bboxes.append(clustered_bbox)
                
                # If both bounding boxes are valid, choose the bigger one to keep
                elif pfcn_valid and clustered_valid:
                    if pfcn_bbox.volume() > clustered_bbox.volume():
                        final_bboxes.append(pfcn_bbox)
                    elif pfcn_bbox.volume() < clustered_bbox.volume():
                        final_bboxes.append(clustered_bbox)

        # Visualize the final bounding box selection
        final_bboxes.extend([full_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])
        o3d.visualization.draw_geometries(final_bboxes)  
