import yaml
import numpy as np
import open3d as o3d

from scipy.special import softmax

if __name__ == "__main__":
    dataset = "siemens"

    if dataset == "siemens":
        # Load inference results
        inf_path = "/Users/Mohamed/Documents/hsd/sphereformer/output/test/lidar_tele_15_1651752733162723670_norm_v3.npy"
        inf = np.load(inf_path)

        inf = softmax(inf, axis=1)

        # Load original point cloud
        pc_path = "/Users/Mohamed/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_tele_15/1651752733162723670.npy"
        pc = np.load(pc_path)
        pc = pc[pc[:, 2] > -1.3]
        points = pc[:, :3]

    elif dataset == "semantic_kitti":
        inf_path = "/Users/Mohamed/Documents/hsd/sphereformer/output/SemanticKITTI/000000.npy"
        inf = np.load(inf_path)

        pc_path = "/Users/Mohamed/Documents/hsd/data/00/velodyne/000000.bin"
        pc = np.fromfile(pc_path, dtype=np.float32).reshape((-1, 4))
        points = pc[:, :3]

    # Interpret inference results
    labels = np.argmax(inf, axis=1)

    # Load label colours
    yaml_file = "/Users/Mohamed/Documents/hsd/sphereformer/SphereFormer/util/semantic-kitti.yaml"
        
    with open(yaml_file, 'r') as f:
        sem_kitti_yaml = yaml.safe_load(f)
    
    learning_map_inv = sem_kitti_yaml["learning_map_inv"]
    color_map = sem_kitti_yaml["color_map"] # bgr format
    
    pcd_list = []
    for i in range(1, 20):
        pcd = o3d.geometry.PointCloud()

        mask = labels == i
        pcd_pts = points[mask]
        pcd.points = o3d.utility.Vector3dVector(pcd_pts)

        pcd_list.append(pcd)

        rgb = np.flip(np.array(color_map[learning_map_inv[i]]) /255.0)
        pcd.paint_uniform_color(rgb)

    o3d.visualization.draw_geometries(pcd_list)
