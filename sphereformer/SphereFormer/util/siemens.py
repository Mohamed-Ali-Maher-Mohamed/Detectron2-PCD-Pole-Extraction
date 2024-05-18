import os
import torch
import numpy as np

from util.data_util import data_prepare


class SiemensDataset(torch.utils.data.Dataset):
    def __init__(self,
        data_path,
        voxel_size=[0.1, 0.1, 0.1],
        split='test',
        voxel_max=None,
        xyz_norm=False
    ):
        super().__init__()
        self.data_path = data_path

        if isinstance(voxel_size, list):
            voxel_size = np.array(voxel_size).astype(np.float32)
        self.voxel_size = voxel_size

        self.split = split
        self.voxel_max = voxel_max
        self.xyz_norm = xyz_norm

        # Get list of filenames
        self.files = sorted(os.listdir(data_path))
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return self.get_single_sample(index)
    
    def get_single_sample(self, index, vote_idx=0):
        # Get path to file
        file_path = os.path.join(self.data_path, self.files[index])

        # Load raw point cloud data (x, y, z, intensity, tag, line)
        points = np.load(file_path).reshape((-1, 6))

        feats = points[:, :4]
        xyz = points[:, :3]
        labels_in = np.zeros(points.shape[0]).astype(np.uint8)

        coords, xyz, feats, labels, inds_reconstruct = data_prepare(xyz, feats, labels_in, self.split, self.voxel_size, self.voxel_max, None, self.xyz_norm)
        
        return coords, xyz, feats, labels, inds_reconstruct, self.files[index]
