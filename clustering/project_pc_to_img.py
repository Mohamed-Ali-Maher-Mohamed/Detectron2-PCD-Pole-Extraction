import os
import tarfile
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from pathlib import Path

from rosbags.typesys import get_types_from_msg, register_types
from rosbags.typesys.base import TypesysError
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr


# function for registering message types from message definitions tarfile
def rosbags_register_types_from_messages_tar(path):
    msg_defs = []

    with open(path, "rb") as file_:
        tar_file_object = tarfile.open(mode="r|*", fileobj=file_)
        while True:
            member = tar_file_object.next()
            if not member:
                break
            if member.size == 0:
                continue
            extracted_file_obj = tar_file_object.extractfile(member)

            data = extracted_file_obj.read()
            msg_def = data.decode("utf-8")
            msg_defs.append((member.name, msg_def))

    for name, msg_def in msg_defs:
        name, _ = name.split(".")
        try:
            register_types(get_types_from_msg(msg_def, name))
        except TypesysError as exc:
            pass

    
def get_extrinsic_matrix(topic, bag_path):
    with Reader(bag_path) as reader:
        # get the connection(s) for the correct topic
        connections = [con for con in reader.connections if con.topic == topic]
        
        # total number of messages
        # print("Total Messages:", next(reader.messages(connections=connections))[0].msgcount)
        
        # iterate all messages
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
            # print(i, datetime.fromtimestamp(timestamp * 1e-9))
            
            # deserialize the message
            msg = deserialize_cdr(rawdata, connection.msgtype)

            T = msg.extrinsic_calibration.transformation
            ext_mat = np.array([[T.rx0, T.ry0, T.rz0, T.tx],
                                  [T.rx1, T.ry1, T.rz1, T.ty],
                                  [T.rx2, T.ry2, T.rz2, T.tz],
                                  [0, 0, 0, 1]])
            
            return ext_mat
    

def get_intrinsic_matrix(topic, bag_path):
    with Reader(bag_path) as reader:
        # get the connection(s) for the correct topic
        connections = [con for con in reader.connections if con.topic == topic]
        
        # total number of messages
        # print("Total Messages:", next(reader.messages(connections=connections))[0].msgcount)
        
        # iterate all messages
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
            # print(i, datetime.fromtimestamp(timestamp * 1e-9))
            
            # deserialize the message
            msg = deserialize_cdr(rawdata, connection.msgtype)

            T = msg.camera_calibration
            int_mat = np.array([[T.fx, T.skew, T.cx],
                                  [0, T.fy, T.cy],
                                  [0, 0, 1]])
            
            return int_mat
    
def project_pc_to_img(pc, img, lidar_ext_mat, cam_ext_mat, cam_int_mat):
    # Grab xyz and form homogenous coordinates
    points = pc[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])

    points = np.hstack((points, np.ones((points.shape[0], 1))))

    # Transform from lidar frame to world frame
    points = np.dot((lidar_ext_mat), points.T).T

    # Transform from world frame to camera frame
    points = np.dot(np.linalg.inv(cam_ext_mat), points.T).T

    # Tranform to image frame (3D to 2D projection)
    points = np.dot(cam_int_mat, points[:, :3].T).T

    points = np.divide(points, np.vstack((points[:, 2], points[:, 2], points[:, 2])).T)

    points = points.astype(int)
    full_points = points.copy()
    points = points[points[:, 0] < img.shape[0]]
    points = points[points[:, 0] >= 0]
    points = points[points[:, 1] < img.shape[1]]
    points - points[points[:, 1] >= 0]

    return full_points, points


if __name__ == "__main__":
    # Path to datasets
    pc_dir = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/lidar_horizon"
    img_dir = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11/cam_imgs"

    pc_files = (os.listdir(pc_dir))
    img_files = (os.listdir(img_dir))

    img_timestamps = np.array([int(f[:-4]) for f in img_files])

    # Load bag file
    bag_path = "/Users/kelly/Documents/hsd/data/od_recording_2022_05_05-12_12_11"
    rosbags_register_types_from_messages_tar(Path(bag_path) / "attachments/message_definitions.tar")

    # Read bag file for extrinsic and intrinsic matrices
    lidar_topic = "/lidar_horizon/lidar/data_cal"
    lidar_ext_mat = get_extrinsic_matrix(lidar_topic, bag_path)

    cam_topic = "/camera_0/data_cal"
    cam_ext_mat = get_extrinsic_matrix(cam_topic, bag_path)
    cam_int_mat = get_intrinsic_matrix(cam_topic, bag_path)

    # Iterate through point clouds. We want to assign labels to point clouds
    for f in pc_files:
        # Find the closest img timestamp to the given pc
        diff = np.abs(img_timestamps - np.repeat(int(f[:-4]), img_timestamps.shape))
        img_idx = np.argmin(diff)

        # Load pc and image now
        pc = np.load(os.path.join(pc_dir, f))
        img = plt.imread(os.path.join(img_dir, img_files[img_idx]))

        print("LiDAR File:", f)
        print("Image File:", img_files[img_idx])

        all_points, points = project_pc_to_img(pc, img, lidar_ext_mat, cam_ext_mat, cam_int_mat)

        implot = plt.imshow(img)
        plt.scatter(points[:, 0], points[:, 1], marker='.', color='red', alpha=1, s=0.5)
        plt.show()

        # break
        