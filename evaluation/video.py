"""Render a summary video as shown on the project page."""
import math

import numpy as np
import torch
import os

from PIL import Image
#im = Image.fromarray(arr)
#im.save("your_file.jpeg")
from . import segmentation, utils


def render(dataset, model, sample_id=None, n_images=20):
    """
    Render a video for a dataset and model.
    If a sample_id is selected, then the view is fixed and images
    are rendered for a specific viewpoint over a timerange (the bottom part
    of the summary video on the project page). Otherwise, images are rendered
    for multiple viewpoints (the top part of the summary video).
    """

    ims = {}

    keys = [
        "mask_pers",
        "mask_tran",
        "mask_pred",
        "im_tran",
        "im_stat",
        "im_pred",
        "im_pers",
        "im_targ",
    ]

    if n_images > len(dataset.img_ids) or n_images == 0:
        n_images = len(dataset.img_ids)

    for i in utils.tqdm(dataset.img_ids[:: math.ceil(len(dataset.img_ids) / n_images)]):
        if sample_id is not None:
            j = sample_id
        else:
            j = i
        timestep = i
        with torch.no_grad():
            x = segmentation.evaluate_sample(
                dataset, j, t=timestep, model=model, visualise=False
            )
            ims[i] = {k: x[k] for k in x if k in keys}
    return ims

def render_separate_imgs(dataset, model,root,sample_id=None, n_images=20):
    """
    Render a video for a dataset and model.
    If a sample_id is selected, then the view is fixed and images
    are rendered for a specific viewpoint over a timerange (the bottom part
    of the summary video on the project page). Otherwise, images are rendered
    for multiple viewpoints (the top part of the summary video).
    """

    root_static = os.path.join(root, "static")
    root_foreground = os.path.join(root, "foreground")
    root_actor = os.path.join(root, "actor")
    root_pred = os.path.join(root, "pred")
    root_targ = os.path.join(root, "target")

    os.makedirs(root_static)
    os.makedirs(root_foreground)
    os.makedirs(root_actor)
    os.makedirs(root_pred)
    os.makedirs(root_targ)

    psnr = {}

    keys = [
        "mask_pers",
        "mask_tran",
        "mask_pred",
        "im_tran",
        "im_stat",
        "im_pred",
        "im_pers",
        "im_targ",
    ]

    if n_images > len(dataset.img_ids) or n_images == 0:
        n_images = len(dataset.img_ids)

    for i in utils.tqdm(dataset.img_ids[:: math.ceil(len(dataset.img_ids) / n_images)]):
        if sample_id is not None:
            j = sample_id
        else:
            j = i
        timestep = i
        with torch.no_grad():
            x = segmentation.evaluate_sample(
                dataset, j, t=timestep, model=model, visualise=False
            )

            #(x * 255).astype(np.uint8)
            im_static = Image.fromarray((x["im_stat"].numpy()*255).astype(np.uint8))
            im_foreground = Image.fromarray((x["im_tran"]*255).astype(np.uint8))
            im_actor = Image.fromarray((x["im_pers"]*255).astype(np.uint8))
            im_pred = Image.fromarray((x["im_pred"].numpy()*255).astype(np.uint8))
            im_targ = Image.fromarray((x["im_targ"].numpy()*255).astype(np.uint8))

            im_static.save(root_static+"/static_"+str(i)+".png")
            im_foreground.save(root_foreground+"/foreground_"+str(i)+".png")
            im_actor.save(root_actor+"/actor_"+str(i)+".png")
            im_pred.save(root_pred+"/pred_"+str(i)+".png")
            im_targ.save(root_targ+"/targ_"+str(i)+".png")

            psnr[i] = {x["psnr"]}
    return psnr

def render_single_view(dataset, model,root,sample_id=None, n_images=20):
    """
    Render a video for a dataset and model.
    If a sample_id is selected, then the view is fixed and images
    are rendered for a specific viewpoint over a timerange (the bottom part
    of the summary video on the project page). Otherwise, images are rendered
    for multiple viewpoints (the top part of the summary video).
    """

    root_static = os.path.join(root, "static")
    
    os.makedirs(root_static)

    psnr = {}

    if n_images > len(dataset.img_ids) or n_images == 0:
        n_images = len(dataset.img_ids)

    #i = dataset.img_ids[:: math.ceil(len(dataset.img_ids) / n_images)]
    i = dataset.img_ids[:: math.ceil(len(dataset.img_ids) / n_images)]
    if sample_id is not None:
        j = sample_id
    else:
        j = i
    timestep = i
    with torch.no_grad():
        x = segmentation.evaluate_sample(
            dataset, j, t=timestep, model=model, visualise=False
        )
        #(x * 255).astype(np.uint8)
        im_static = Image.fromarray((x["im_stat"].numpy()*255).astype(np.uint8))
        
        im_static.save(root_static+"/static_"+str(i)+".png")
        
        psnr[i] = {x["psnr"]}
    return psnr



def cat_sample(top, bot):
    """Concatenate images from the top and bottom part of the summary video."""
    keys = ["im_targ", "im_pred", "im_stat", "im_tran", "im_pers"]
    top = np.concatenate([(top[k]) for k in keys], axis=1)
    bot = np.concatenate([(bot[k]) for k in keys], axis=1)
    bot[
        :,
        : bot.shape[1] // len(keys),  # black background in first column
    ] = (0, 0, 0)
    z = np.concatenate([top, bot], axis=0)
    return z


def save_to_cache(vid, sid, root, top=None, bot=None):
    """Save the images for rendering the video."""
    if top is not None:
        p = f"{root}/images-{sid}-top.pt"
        if os.path.exists(p):
            print("images exist, aborting.")
            return
        torch.save(top, p)
    if bot is not None:
        p = f"{root}/images-{sid}-bot.pt"
        if os.path.exists(p):
            print("images exist, aborting.")
            return
        torch.save(bot, p)


def load_from_cache(vid, sid, root, version=0):
    """Load the images for rendering the video."""
    path_top = f"{root}/images-{sid}-top.pt"
    path_bot = f"{root}/images-{sid}-bot.pt"
    top = torch.load(path_top)
    bot = torch.load(path_bot)
    return top, bot


def convert_rgb(im):
    im[im > 1] = 1
    im = (im * 255).astype(np.uint8)
    return im
