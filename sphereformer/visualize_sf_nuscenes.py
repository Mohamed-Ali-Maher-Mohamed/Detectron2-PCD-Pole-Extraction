import yaml
import numpy as np
import open3d as o3d


if __name__ == '__main__':
    # Load inference results
    inf_path = "/Users/kelly/Documents/hsd/sphereformer/output/test_nuscenes/lidar_tele_15_1651752733162723670_norm_v3.npy"
    inf = np.load(inf_path)

     # Load original point cloud
    pc_path = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_tele_15/1651752733162723670.npy"
    pc = np.load(pc_path)
    pc = pc[pc[:, 2] > -1.3]
    points = pc[:, :3]

    # Interpret inference results
    labels = np.argmax(inf, axis=1)

    # Load label colours
    yaml_file = "/Users/kelly/Documents/hsd/sphereformer/SphereFormer/util/semantic-kitti.yaml"
        
    with open(yaml_file, 'r') as f:
        sem_kitti_yaml = yaml.safe_load(f)
    
    learning_map_inv = sem_kitti_yaml["learning_map_inv"]
    color_map = sem_kitti_yaml["color_map"] # bgr format

    pcd_list = []
    for i in range(1, 16):
        pcd = o3d.geometry.PointCloud()

        mask = labels == i
        pcd_pts = points[mask]
        pcd.points = o3d.utility.Vector3dVector(pcd_pts)

        pcd_list.append(pcd)

        rgb = np.flip(np.array(color_map[learning_map_inv[i]]) /255.0)
        pcd.paint_uniform_color(rgb)

    pcd_list.append(o3d.geometry.TriangleMesh.create_coordinate_frame(10))
    o3d.visualization.draw_geometries(pcd_list)
