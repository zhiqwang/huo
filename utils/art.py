import os
import numpy as np
import torch
import torch.nn.functional as F

# algebraic reconstruction technique
def art(sinogram, img_grid, detr_end, gantry_coor_x, gantry_coor_y, gantry_view, param):

    img_coor_y, img_coor_x = torch.meshgrid(img_grid, img_grid)
    sinogram = sinogram.unsqueeze(0).unsqueeze(0)
    img = torch.zeros(param.img_pixels, param.img_pixels)

    indices = torch.randperm(len(gantry_view))
    for i in indices:
        view = gantry_view[i]
        res = sinogram[:, :, :, i] - forward_propagation(img, gantry_coor_x, gantry_coor_y, view, param)
        res /= param.img_len
        img += backward_propagation(res, img_coor_x, img_coor_y, detr_end, view, param)
        img = torch.clamp(img, min=0)
    # img = img.squeeze(0).squeeze(0)

    return img

# doing CT scaning
def scan(img, gantry_coor_x, gantry_coor_y, gantry_view, param):

    sinogram = torch.zeros(param.detr_num, len(gantry_view))
    sinogram = sinogram.unsqueeze(0).unsqueeze(0)
    # scaning per angle
    for i, view in enumerate(gantry_view):
        sinogram[:, :, :, i] = forward_propagation(img, gantry_coor_x, gantry_coor_y, view, param)
    sinogram = sinogram.squeeze(0).squeeze(0)

    return sinogram

# forward propagation per angle
def forward_propagation(img, gantry_coor_x, gantry_coor_y, view, param):
    # counter clockwise
    # Set angle
    img = img.unsqueeze(0).unsqueeze(0)
    view = torch.tensor([view * np.pi / 180.])

    gantry_rot_x = gantry_coor_x * torch.cos(view) - gantry_coor_y * torch.sin(view)
    gantry_rot_y = gantry_coor_x * torch.sin(view) + gantry_coor_y * torch.cos(view)

    gantry_rot_x = gantry_rot_x.unsqueeze(0).unsqueeze(3)
    gantry_rot_y = gantry_rot_y.unsqueeze(0).unsqueeze(3)
    samples = torch.cat([gantry_rot_x, gantry_rot_y], 3)
    img_gantry_interp = F.grid_sample(img, samples)
    img_gantry_interp = img_gantry_interp.squeeze(0).squeeze(0)
    lat_step = param.img_len / param.img_pixels / param.lat_sampling
    sinogram = lat_step * torch.sum(img_gantry_interp, dim=0)
    return sinogram

# backward propagation per angle
def backward_propagation(sinogram, img_coor_x, img_coor_y, detr_end, view, param):
    # pre processing
    sinogram = sinogram.unsqueeze(3)
    # set angle
    view = torch.tensor([view * np.pi / 180.])

    # clockwise
    img_rot_x = img_coor_x * torch.cos(view) + img_coor_y * torch.sin(view)
    img_rot_y = - img_coor_x * torch.sin(view) + img_coor_y * torch.cos(view)

    img_rot_y = img_rot_y + param.sod
    samples = param.sdd * img_rot_x / img_rot_y
    samples /= detr_end
    samples = samples.reshape(-1,1)
    samples = samples.unsqueeze(2).unsqueeze(0)
    samples = torch.cat([torch.zeros_like(samples), samples], 3)
    # print('res: {}'.format(sinogram.shape))
    # print('samples: {}'.format(samples.shape))

    img = F.grid_sample(sinogram, samples)
    img = img.squeeze(0).squeeze(0)
    img = img.reshape(param.img_pixels, param.img_pixels)

    return img
