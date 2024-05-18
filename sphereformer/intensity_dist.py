import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Load the point clouds 
    # siemens_path = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_tele_15/1651752733162723670.npy"
    siemens_path = '/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_horizon/1651752733162245419.npy'
    kitti_path = "/Users/kelly/Documents/hsd/data/00/velodyne/000000.bin"

    siemens_pc = np.load(siemens_path)
    kitti_pc = pc = np.fromfile(kitti_path, dtype=np.float32).reshape((-1, 4))

    siemens_points = siemens_pc[:, :3]
    kitti_points = kitti_pc[:, :3]

    siemens_intensity = siemens_pc[:, 3]
    kitti_intensity = kitti_pc[:, 3]

    siemens_pcd = o3d.geometry.PointCloud()
    siemens_pcd.points = o3d.utility.Vector3dVector(siemens_points)
    siemens_pcd = siemens_pcd.scale(1/5, center=[0, 0, 0])
    siemens_pcd = siemens_pcd.translate([0, 0, -1.5])
    o3d.visualization.draw_geometries([siemens_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])

    kitti_pcd = o3d.geometry.PointCloud()
    kitti_pcd.points  = o3d.utility.Vector3dVector(kitti_points)
    o3d.visualization.draw_geometries([kitti_pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])

    siemens_intensity_norm = (siemens_intensity-np.min(siemens_intensity))/(np.max(siemens_intensity)-np.min(siemens_intensity))

    # Plot intensity distributions
    plt.hist(siemens_intensity_norm, bins=100, label='siemens')
    plt.legend()
    plt.show()

    plt.hist(kitti_intensity, bins=100, label='semantic_kitti', color='orange')
    plt.legend()
    plt.show()

    # Find Euclidean distance of each point from world frame
    siemens_dist = np.linalg.norm(siemens_points, axis=1)
    kitti_dist = np.linalg.norm(kitti_points, axis=1)

    plt.scatter(siemens_dist, siemens_intensity_norm, s=2)
    plt.xlabel('siemens_dist')
    plt.ylabel('siemens_intensity')
    plt.show()

    plt.scatter(kitti_dist, kitti_intensity, s=1, c='orange')
    plt.xlabel('kitti_dist')
    plt.ylabel('kitti_intensity')
    plt.show()
    