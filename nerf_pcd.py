
# Load Nerf Model
import argparse
import os
import open3d as o3d
import numpy as np

import evaluation
import utils
import json
from dataset import SAMPLE_IDS, VIDEO_IDS, EPICDiff, MaskLoader

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader,Dataset


class PCDDataset(Dataset):
    def __init__(self, path):  # Pass any necessary parameters
        self.pcd = o3d.io.read_point_cloud(path)

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
    dataset = EPICDiff(args.vid, root=args.root_data)
    #Get Pointcloud
    ply_dataset = PCDDataset(args.ply_path)
    data_loader = DataLoader(dataset=ply_dataset, batch_size=2*1024, shuffle=False)

    
    #args.path = path del ckpt
    model = utils.init_model(args.path, dataset)
    model.eval()

    # update parameters of loaded models
    model.hparams["suppress_person"] = False
    model.hparams["inference"] = True

    return model, dataset, ply_dataset, data_loader


def render_separate_video(args, model, dataset, root, save_cache=False):
    """Render  separate frames."""
    root = os.path.join(root, "decomposition")
    os.makedirs(root)

    sid = SAMPLE_IDS[args.vid]

    psnr = evaluation.video.render_separate_imgs(
        dataset, model,root,n_images=args.summary_n_samples
    )

    # Serializing json
    json_object = json.dumps(psnr, indent=1)
    json_path = os.path.join(root, "psnr.json")
    # Writing to meta.json
    with open(json_path, "w") as outfile:
        outfile.write(json_object)
    
    

def run(model,dataloader):
    new_pcd = np.zeros((1,7))
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_xyz, batch_rgb = batch
            input_dir = (torch.tensor([0,1,0],dtype=float)).repeat(batch_xyz.shape[0],1)
            batch_xyz_embed = model.embedding_xyz(batch_xyz)
            dir_embed =  model.embedding_dir(input_dir)
            nerf_in = torch.cat((batch_xyz_embed,dir_embed),dim=1)
            nerf_in = (nerf_in.to("cuda:0")).float()
            # Batch x ( rgb:3 + sigma:1 )
            pred = model.nerf_coarse(nerf_in,output_dynamic=False)
            
            tmp_pcd = np.concatenate((batch_xyz,pred.cpu().numpy()),axis=1)
            new_pcd = np.concatenate((new_pcd,tmp_pcd))

    
    new_pcd = np.delete(new_pcd,0,axis=0) #remove first row
    np.save("pcd_P01_01.npy",new_pcd)
    #save as .ply
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(new_pcd[:,0:3])
    ply.colors = o3d.utility.Vector3dVector(new_pcd[:,3:6])
    o3d.io.write_point_cloud("pcd_P01_01.ply", ply)
    return 1
    



if __name__ == "__main__":
    args = parse_args()
    
    model, dataset, ply_dataset,ply_dataloader = init(args)
    root = os.path.join("pcds", args.exp, args.vid)
    run(model,ply_dataloader)

#Load COLMAP PCD

