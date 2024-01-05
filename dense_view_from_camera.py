import open3d as o3d


import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torch
import matplotlib.pyplot as plt
import json

import cv2
from tqdm import tqdm


from datetime import datetime
import copy
import argparse
from scipy.spatial.distance import cdist

def load_meta(root, name="meta.json"):
    """Load meta information per scene and frame (nears, fars, poses etc.)."""
    path = os.path.join(root, name)
    with open(path, "r") as fp:
        ds = json.load(fp)
    for k in ["nears", "fars", "images", "poses"]:
        ds[k] = {int(i): ds[k][i] for i in ds[k]}
        if k == "poses":
            ds[k] = {i: np.array(ds[k][i]) for i in ds[k]}
    ds["intrinsics"] = np.array(ds["intrinsics"])
    return ds

class EPICDiff(Dataset):
    def __init__(self, vid, root="data/EPIC-Diff", split=None):

        self.root = os.path.join(root, vid)
        self.vid = vid
        self.img_w = 228
        self.img_h = 128
        self.split = split
        self.val_num = 1
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), #TODO
            torchvision.transforms.Resize((self.img_h,self.img_w))
        ])
        self.init_meta()

    def imshow(self, index):
        plt.imshow(self.imread(index))
        plt.axis("off")
        plt.show()

    def imread(self, index):
        return plt.imread(os.path.join(self.root, "frames", self.image_paths[index]))

    def x2im(self, x, type_="np"):
        """Convert numpy or torch tensor to numpy or torch 'image'."""
        w = self.img_w
        h = self.img_h
        if len(x.shape) == 2 and x.shape[1] == 3:
            x = x.reshape(h, w, 3)
        else:
            x = x.reshape(h, w)
        if type(x) == torch.Tensor:
            x = x.detach().cpu()
            if type_ == "np":
                x = x.numpy()
        elif type(x) == np.array:
            if type_ == "pt":
                x = torch.from_numpy(x)
        return x

    def rays_per_image(self, idx, pose=None):
        """Return sample with rays, frame index etc."""
        sample = {}
        if pose is None:
            sample["c2w"] = c2w = torch.FloatTensor(self.poses_dict[idx])
        else:
            sample["c2w"] = c2w = pose

        sample["im_path"] = self.image_paths[idx]

        img = Image.open(os.path.join(self.root, "frames", self.image_paths[idx]))
        img_w, img_h = img.size
        img = self.transform(img)  # (3, h, w)
        _,img_h,img_w = img.size() #TODO Check thissss
        img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB

        directions = get_ray_directions(img_h, img_w, self.K)
        rays_o, rays_d = get_rays(directions, c2w)

        c2c = torch.zeros(3, 4).to(c2w.device)
        c2c[:3, :3] = torch.eye(3, 3).to(c2w.device)
        rays_o_c, rays_d_c = get_rays(directions, c2c)

        rays_t = idx * torch.ones(len(rays_o), 1).long()

        rays = torch.cat(
            [
                rays_o,
                rays_d,
                self.nears[idx] * torch.ones_like(rays_o[:, :1]),
                self.fars[idx] * torch.ones_like(rays_o[:, :1]),
                rays_o_c,
                rays_d_c,
            ],
            1,
        )

        sample["rays"] = rays
        sample["img_wh"] = torch.LongTensor([img_w, img_h])
        sample["ts"] = rays_t
        sample["rgbs"] = img

        return sample

    def init_meta(self):
        """Load meta information, e.g. intrinsics, train, test, val split etc."""
        meta = load_meta(self.root)
        self.img_ids = meta["ids_all"]
        self.img_ids_train = meta["ids_train"]
        self.img_ids_test = meta["ids_test"]
        self.img_ids_val = meta["ids_val"]
        self.poses_dict = meta["poses"]
        self.nears = meta["nears"]
        self.fars = meta["fars"]
        self.image_paths = meta["images"]
        self.K = meta["intrinsics"]

        if self.split == "train":
            # create buffer of all rays and rgb data
            self.rays = []
            self.rgbs = []
            self.ts = []

            for idx in self.img_ids_train:
                sample = self.rays_per_image(idx)
                self.rgbs += [sample["rgbs"]]
                self.rays += [sample["rays"]]
                self.ts += [sample["ts"]]

            self.rays = torch.cat(self.rays, 0)  # ((N_images-1)*h*w, 8)
            self.rgbs = torch.cat(self.rgbs, 0)  # ((N_images-1)*h*w, 3)
            self.ts = torch.cat(self.ts, 0)

    def __len__(self):
        if self.split == "train":
            # rays are stored concatenated
            return len(self.rays)
        if self.split == "val":
            # evaluate only one image, sampled from val img ids
            return 1
        else:
            # choose any image index
            return max(self.img_ids)

    def __getitem__(self, idx, pose=None):

        if self.split == "train":
            # samples selected from prefetched train data
            sample = {
                "rays": self.rays[idx],
                "ts": self.ts[idx, 0].long(),
                "rgbs": self.rgbs[idx],
            }

        elif self.split == "val":
            # for tuning hyperparameters, tensorboard samples
            idx = random.choice(self.img_ids_val)
            sample = self.rays_per_image(idx, pose)

        elif self.split == "test":
            # evaluating according to table in paper, chosen index must be in test ids
            assert idx in self.img_ids_test
            sample = self.rays_per_image(idx, pose)

        else:
            # for arbitrary samples, e.g. summary video when rendering over all images
            sample = self.rays_per_image(idx, pose)

        return sample
import numpy as np

def points_visible_to_camera(point_cloud, intrinsic_matrix, extrinsic_matrix,width=228,height=128):
    """
    Filter points in a point cloud that are visible to a camera.

    Parameters:
    - point_cloud (numpy array): Array of shape (N, 3) representing the 3D points in world coordinates.
    - intrinsic_matrix (numpy array): 3x3 intrinsic matrix representing the camera intrinsics.
    - extrinsic_matrix (numpy array): 4x4 extrinsic matrix representing the camera pose.

    Returns:
    - numpy array: Array of shape (M, 3) representing the visible points.
    """
    # Full projection matrix
    projection_matrix = np.dot(intrinsic_matrix, extrinsic_matrix)

    # Homogeneous coordinates for 3D points
    point_cloud_homogeneous = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))

    # Project points to 2D image coordinates
    image_points_homogeneous = np.dot(projection_matrix, point_cloud_homogeneous.T).T
    image_points = image_points_homogeneous[:, :2] / image_points_homogeneous[:, 2][:, None]

    # Filter points within image bounds
    visible_points = point_cloud[(0 <= image_points[:, 0] < width) & (0 <= image_points[:, 1] < height)]

    return visible_points

def points_visible_to_camera_2(point_cloud, intrinsic_matrix, extrinsic_matrix,th=0.8,width=228,height=128):
    #translate
    pcd_camera = np.asarray(point_cloud.points) - extrinsic_matrix[:3,3]
    #dot product
    pcd_norms = np.linalg.norm(pcd_camera, axis=1, keepdims=True)
    pcd_normalized = pcd_camera / pcd_norms

    view_direction = -extrinsic_matrix[:,2]
    alphas = np.dot(pcd_normalized,view_direction/(np.linalg.norm(view_direction,keepdims=True)))

    indices = np.where(alphas > th)[0]
    return indices

def visibility(pcd,camera_poses,vid):
    visible = {}
    for k in tqdm(range(0, 1000, 5)):
        extrinsic_camera = camera_poses[k]
        camera_intrinsic = ed.K
        indxs = points_visible_to_camera_2(pcd,camera_intrinsic,extrinsic_camera)
        visible[k] = indxs.tolist()
    
    file_path = "visible_"+vid+".json"
    with open(file_path, 'w') as json_file:
        json.dump(visible, json_file)
    return visible

path_pcd = "/scratch/fborgna/EPIC_F_rec/P01_01/dense/fused.ply"
pcd  = o3d.io.read_point_cloud(path_pcd)

vid = "P01_01"
ed = EPICDiff(vid, root="data/Epic_converted")

camera_poses = ed.poses_dict
camera_intrinsic = ed.K

extrinsic_camera = camera_poses[285]
indxs = points_visible_to_camera_2(pcd,camera_intrinsic,extrinsic_camera)

visibility(pcd,camera_poses,vid)