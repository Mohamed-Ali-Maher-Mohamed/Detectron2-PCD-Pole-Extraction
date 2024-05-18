import numpy as np
import open3d as o3d

if __name__ == '__main__':
    data_path = '/Users/Mohamed/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_tele_15/1651752733162723670.npy'
    out_path = '/Users/Mohamed/Documents/hsd/data/lidar_tele_15_1651752733162723670_norm_v3.npy'

    pc = np.load(data_path)

    # Remove ground plane
    mask = pc[:, 2] > -1.3
    pc = pc[mask]

    # Normalize intensity values
    pc[:, 3] = (pc[:, 3]-np.min(pc[:, 3]))/(np.max(pc[:, 3])-np.min(pc[:, 3]))

    # Scale and translate coordinate frame
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    pcd = pcd.scale(1/5, center=[0, 0, 0])
    pcd = pcd.translate([0, 0, -1.5])
    o3d.visualization.draw_geometries([pcd, o3d.geometry.TriangleMesh.create_coordinate_frame(10)])

    pc[:, :3] = np.array(pcd.points)

    np.save(out_path, pc)
    
