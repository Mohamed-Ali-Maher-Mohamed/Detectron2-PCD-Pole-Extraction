import os
from pathlib import Path
from datetime import datetime
import tarfile

import numpy as np
from PIL import Image
import cv2
from rosbags.typesys import get_types_from_msg, register_types
from rosbags.typesys.base import TypesysError
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr

from pydamsy.documents import DocumentsClient
from pydamsy.aistore.document import Document, File
from pydamsy.aistore.session import Session
from pydamsy.aistore.pipelines import PipelineRun, ProblemType, PipelineStepSection, PipelineStep, Evaluation

import open3d as o3d

from point_cloud2 import *


# function for downloading a given ros2bag
def download_ros2bag(session, doc_id, output_dir):
    ros2bag_document = Document.from_existing(session, "ros2_bag_v4", doc_id)
    bag_directory = Path(output_dir) / ros2bag_document.raw_document["name"]

    bag_directory.mkdir(exist_ok=True)

    attachments_directory = bag_directory / "attachments"

    file_edges = ros2bag_document.edges(doctype="file")
    for file_edge in file_edges:
        file_doc = File.from_existing(session, file_edge["target"]["id"])
        if file_edge["name"] in ["db", "metadata"]:
            file_doc.download_file(base_directory=bag_directory, print_progress=True)
        elif file_edge["name"] == "attachment":
            attachments_directory.mkdir(exist_ok=True)
            file_doc.download_file(base_directory=attachments_directory, print_progress=True)

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

def save_lidar_pcds(topic, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True) 

    with Reader(bag_path) as reader:
        # get the connection(s) for the correct topic
        connections = [con for con in reader.connections if con.topic == topic]
        
        # total number of messages
        print("Total Messages:", next(reader.messages(connections=connections))[0].msgcount)
        
        # iterate all messages
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
            print("LiDAR:", i, datetime.fromtimestamp(timestamp * 1e-9))
            
            # deserialize the message
            msg = deserialize_cdr(rawdata, connection.msgtype)

            pts = [[p[0], p[1], p[2], p[3], p[4], p[5]] for p in read_points(msg, field_names = ("x", "y", "z", "intensity", "tag", "line"), skip_nans=True)]
            pts = np.array(pts)

            np.save(Path(output_dir) / f"{timestamp}.npy", pts)
            
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
            # o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh.create_coordinate_frame(), pcd])


def save_camera_imgs(topic, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True) 

    with Reader(bag_path) as reader:
        # get the connection(s) for the correct topic
        connections = [con for con in reader.connections if con.topic == topic]
        
        # total number of messages
        print("Total Messages:", next(reader.messages(connections=connections))[0].msgcount)
        
        # iterate all messages
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections=connections)):
            print("Camera:", i, datetime.fromtimestamp(timestamp * 1e-9))
            
            # deserialize the message
            msg = deserialize_cdr(rawdata, connection.msgtype)
            
            # get image data
            img_bayer = np.reshape(msg.data, (int(msg.height), int(msg.width)))  # extract image from message (stored in msg.payload)
            img_rgb = cv2.cvtColor(img_bayer, cv2.COLOR_BayerBG2RGB)  # convert bayer filtered data to RGB
            
            # save image as png
            img = Image.fromarray(img_rgb, "RGB")
            img.save(Path(output_dir) / f"{timestamp}.png")


if __name__ == '__main__':
    # ------ DOWNLOADING DATA ------
    os.environ["DMS_API_TOKEN"] = "eyJhbGciOiJSUzI1NiIsImtpZCI6ImYzNjBhMTJmLWY5ZjktNGRkOC1hYTc2LTI1NWZlZmRmZGM2MCIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL3VzZXJzLmRhbXN5LnNpZW1lbnMuY2xvdWQiLCJzdWIiOiJYUzAwOThHTyIsImV4cCI6MTY4ODY4MDc5OSwiYXBpX3Rva2VuIjoiM2Q4NzNjYjEtMThkZC00ZjA1LWJmZjEtZmE2NzNiZWI5OTliIn0.jWqo6WFDjUQFcEUi8ARslOc7kR6zezdRGzXOG5UVO3rYFqLtGlNJeyMcyy39D_KvC3mtJAl-xTr7I29Oh1zRwj7MZSq1kHxQhDujFQN8TLRPMsQW43ZI5L_CoiBLZ32oTRN7OFzVQ2AvFL61YZPNIMt8G29v3Ei7P5c3xfgSli323eekkbX4Bgi_k57ZLn2l7Xs2gOVqRady3RkWj42CsAu37GMuDCoZtQ5aFPGYfg-mOYdLr_u5SlMTUAs3SKAb0h4npd_iBND4m1CxjY1iq4cDTYJbnqZzXJo4IOW_oNDaYMwwP4idxhP67ophhPs9tPJoUaXLZyLOmMI6m_bvQA"

    DMS_URL = "https://damsy.siemens.cloud"
    session = Session(DMS_URL, owner="SFETD")

    bag_id = "23aa683b-4f51-43e7-a778-d61128a4a58f"
    download_ros2bag(session, bag_id, ".")
    bag_path = "./data/od_recording_2022_05_05-12_12_11"  # path of the downloaded ros2bag

    rosbags_register_types_from_messages_tar(Path(bag_path) / "attachments/message_definitions.tar")

    # ------ READING LIDAR DATA ------
    lidar_topic = "/lidar_horizon/lidar/data_debug"
    pcd_path = os.path.join(bag_path, "lidar_horizon")

    save_lidar_pcds(lidar_topic, pcd_path)

    # ------ READING CAMERA DATA ------
    camera_topic = "/camera_0/data_cal"  # topic that contains the camera images
    image_path = os.path.join(bag_path, "cam_imgs")  # path where the exported images will be saved to

    camera_topic = "/odai/track_mask"
    image_path = os.path.join(bag_path, "track_seg")

    save_camera_imgs(camera_topic, image_path)
