# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

import numpy as np
import torch
import torch.nn.functional as F


def art(sinogram, img_end, detr_end, gantry_coor_x, gantry_coor_y, gantry_view, param):
    """
    Algebraic reconstruction technique
    """

    sinogram = sinogram.unsqueeze(0).unsqueeze(0)
    img = torch.zeros(param.img_pixels, param.img_pixels)

    indices = torch.randperm(len(gantry_view))
    for i in indices:
        view = gantry_view[i]
        res = sinogram[:, :, :, i] - forward_propagation(img, gantry_coor_x, gantry_coor_y, view, param)
        res /= param.img_len
        img += backward_propagation(res, img_end, detr_end, view, param)
        img = torch.clamp(img, min=0)

    return img


def scan(img, gantry_coor_x, gantry_coor_y, gantry_view, param):
    """
    CT scaning
    """

    sinogram = torch.zeros(param.detr_num, len(gantry_view))
    sinogram = sinogram.unsqueeze(0).unsqueeze(0)
    # scaning per angle
    for i, view in enumerate(gantry_view):
        sinogram[:, :, :, i] = forward_propagation(img, gantry_coor_x, gantry_coor_y, view, param)
    sinogram = sinogram.squeeze(0).squeeze(0)

    return sinogram


def forward_propagation(img, gantry_coor_x, gantry_coor_y, view, param):
    """
    Forward propagation per angle
    """
    # rotate counter clockwise
    # Set angle
    img = img.unsqueeze(0).unsqueeze(0)
    view = torch.tensor([view * np.pi / 180.0])

    gantry_rot_x = gantry_coor_x * torch.cos(view) - gantry_coor_y * torch.sin(view)
    gantry_rot_y = gantry_coor_x * torch.sin(view) + gantry_coor_y * torch.cos(view)

    gantry_rot_x = gantry_rot_x.unsqueeze(0).unsqueeze(3)
    gantry_rot_y = gantry_rot_y.unsqueeze(0).unsqueeze(3)
    samples = torch.cat([gantry_rot_x, gantry_rot_y], 3)
    img_gantry_interp = F.grid_sample(img, samples, align_corners=True)
    img_gantry_interp = img_gantry_interp.squeeze(0).squeeze(0)
    lat_step = param.img_len / param.img_pixels / param.lat_sampling
    sinogram = lat_step * torch.sum(img_gantry_interp, dim=0)
    return sinogram


def backward_propagation(sinogram, img_end, detr_end, view, param):
    """
    Backward propagation per angle
    """
    # pre processing
    sinogram = sinogram.unsqueeze(3)
    # set angle
    view = torch.tensor([view * np.pi / 180.0])
    # rotation matrix
    theta = torch.tensor(
        [[torch.cos(view), torch.sin(view), 0], [-torch.sin(view), torch.cos(view), 0]]
    ).unsqueeze(0)

    img_grid_rot = F.affine_grid(theta, (1, 1, param.img_pixels, param.img_pixels)).squeeze(0)
    img_rot_x = img_grid_rot[:, :, 0]
    img_rot_y = img_grid_rot[:, :, 1]

    img_rot_y = img_rot_y + param.sod / img_end
    samples = param.sdd * img_rot_x / img_rot_y
    samples /= detr_end
    samples = samples.reshape(-1, 1)
    samples = samples.unsqueeze(2).unsqueeze(0)
    samples = torch.cat([torch.zeros_like(samples), samples], 3)

    img = F.grid_sample(sinogram, samples, align_corners=True)
    img = img.squeeze(0).squeeze(0)
    img = img.reshape(param.img_pixels, param.img_pixels)

    return img
