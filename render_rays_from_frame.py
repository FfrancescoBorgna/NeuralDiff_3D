
# Load Nerf Model
import argparse
import os
import open3d as o3d
import numpy as np

from collections import defaultdict
import evaluation
import utils
import json
from dataset import SAMPLE_IDS, VIDEO_IDS, EPICDiff, MaskLoader

from model.rendering import render_rays
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader,Dataset


class PCDDataset(Dataset):
    def __init__(self, path,indxs=None):  # Pass any necessary parameters
        pcd = o3d.io.read_point_cloud(path)
        if indxs is not None:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[indxs])
            self.pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[indxs])
        else:
            self.pcd = pcd

    def __len__(self):
        return len(np.asarray(self.pcd.points))

    def __getitem__(self, index):
        sample = self.pcd.points[index]  # Implement how to retrieve a single sample
        color = self.pcd.colors[index]
        return sample,color

def parse_args(path=None, vid=None, exp=None, ply_path=None):

    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, default=path, help="Path to model.")

    parser.add_argument("--vid", type=str, default=vid, help="Video ID of dataset.")

    parser.add_argument("--exp", type=str, default=exp, help="Experiment name.")

    parser.add_argument("--ply_path", type=str, default=ply_path, help="Path to COLMAP pcd.ply")

    parser.add_argument(
        "--root_data", type=str, default="data/EPIC-Diff", help="Root of the dataset."
    )

    parser.add_argument("--is_eval_script", default=True, action="store_true")
    args = parser.parse_args()

    return args


def init(args):
    
    #Get Main dataset
    torch.cuda.empty_cache()
    dataset = EPICDiff(args.vid, root=args.root_data)
    #Get Pointcloud
    #ply_dataset = PCDDataset(args.ply_path)
    #data_loader = DataLoader(dataset=ply_dataset, batch_size=2*1024, shuffle=False)
    ply_dataset = 0
    data_loader = 0
    
    #args.path = path del ckpt
    model = utils.init_model(args.path, dataset)
    model.eval()

    # update parameters of loaded models
    model.hparams["suppress_person"] = False
    model.hparams["inference"] = True

    return model, dataset, ply_dataset, data_loader

def run(model, dataset,indx,test_time=True):
    #
    sample = dataset.rays_per_image(indx)

    perturb = 0 if test_time   else model.hparams.perturb
    noise_std = 0 if test_time  else model.hparams.noise_std
  
    rays = sample['rays']
    ts=sample['ts']
    B = rays.shape[0]

    results = defaultdict(list)
    chunk = 1024
    for i in range(0, B, chunk):
        rendered_ray_chunks = render_rays(
                models=model.models,
                embeddings=model.embeddings,
                rays=rays[i : i + chunk].to("cuda"),
                ts=ts[i : i + chunk].to("cuda"),
                N_samples=model.hparams.N_samples,
                perturb=perturb,
                noise_std=noise_std,
                N_importance=model.hparams.N_importance,
                chunk=chunk,
                hp=model.hparams,
                test_time=test_time,
        )
        for k, v in rendered_ray_chunks.items():
                results[k] += [v]

    for k, v in results.items():
            results[k] = torch.cat(v, 0)
    
    return results


def run2(model, dataset,indx,test_time=True):
    #
    sample = dataset.rays_per_image(indx)

    perturb = 0 if test_time   else model.hparams.perturb
    noise_std = 0 if test_time  else model.hparams.noise_std
  
    results = model.render(sample)

    out = {}
    #interested_keys = ["xyz_fine","xyz_coarse","static_sigmas_coarse","static_sigmas_fine","_rgb_fine_static","rgb_fine_static","static_rgbs_fine","static_rgbs_coarse"]
    interested_keys_coarse = ["xyz_coarse","static_sigmas_coarse"]
    interested_keys_fine = ["xyz_fine","static_sigmas_fine","static_rgbs_fine"]

    print("Writing coarse model")
    file_path_coarse = "coarse.pth"
    for ik in interested_keys_coarse:
        out[ik] = results[ik]
    torch.save(out, file_path_coarse)

    print("Writing fine model")
    out = {}
    file_path_fine = "fine.pth"
    for ik in interested_keys_fine:
        out[ik] = results[ik]
    torch.save(out, file_path_fine)
    #JSON ????????????????????????????????????????????????????
    #for ik in interested_keys:
    #     out[ik] = results[ik].cpu().numpy().tolist()
    #
    #write_json("single_frame_grid.json",out)
    print("Ciao")

def load_json(path):
    # Load the JSON file into a Python object
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data

def write_json(path,data):
    with open(path, 'w') as json_file:
        json.dump(data, json_file)
    return 1
if __name__ == "__main__":
    args = parse_args()
    
    model, dataset, ply_dataset,ply_dataloader = init(args)
    indx = 285
    results = run2( model, dataset,indx)

#Load COLMAP PCD

