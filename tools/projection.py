# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

import argparse

import numpy as np
import torch

from huo.art import art


def main(param):
    # image and gantry coordinate config
    img_step = param.img_len / param.img_pixels  # pixel step
    img_end = (param.img_len - img_step) / 2  # image unit
    detr_step = param.detr_len / param.detr_num  # detector step (longitude step of ray)
    detr_end = (param.detr_len - detr_step) / 2

    # 0 degree for the X ray source is on the top vertical axis,
    # assume the gantry rotate clockwise by default.
    gantry_view = torch.arange(0, 360, step=param.rotate_step)

    # coordinate calculation
    src_coor = (0, -param.sod)

    detr_coor_x = torch.linspace(-detr_end, detr_end, steps=param.detr_num)
    detr_coor_y = torch.zeros_like(detr_coor_x) + param.sdd - param.sod

    # compute coordinate of gantry
    lat_end = param.img_len / 2
    lat_steps = param.lat_sampling * param.img_pixels + 1
    lat_grid = torch.linspace(-lat_end, lat_end, steps=lat_steps).unsqueeze(1)
    # coordinate origin move upward to X-ray source
    lat_grid -= src_coor[1]
    detr_coor_y -= src_coor[1]

    fand = (detr_coor_x**2 + detr_coor_y**2).sqrt()
    gantry_coor_x = lat_grid * detr_coor_x / fand
    gantry_coor_y = lat_grid * detr_coor_y / fand
    # coordinate origin move downward to the center of ratation
    gantry_coor_y += src_coor[1]
    # detr_coor_y += src_coor[1]

    # normalize to between -1 and 1
    gantry_coor_x /= img_end
    gantry_coor_y /= img_end

    # forward
    # img = Image.open('./data/shepp2d.tif')
    # img = np.array(img) / 255.
    # img = torch.from_numpy(img).type(torch.FloatTensor)
    # print('image: {}'.format(img.shape))

    # sinogram = scan(img, gantry_coor_x, gantry_coor_y, gantry_view, param)
    # print('sinogram: {}'.format(sinogram.shape))

    # backward
    sinogram = np.load("./data/sinogram.npy")
    print("sinogram: {}".format(sinogram.shape))
    sinogram = torch.from_numpy(sinogram)
    # reconstruction
    img = art(sinogram, img_end, detr_end, gantry_coor_x, gantry_coor_y, gantry_view, param)
    print(f"image: {img.shape}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Algebraic reconstruction technique")

    # export directory, training and val datasets, test datasets
    parser.add_argument("--img-pixels", default=512, type=int, metavar="NX", help="sample pixels of image")
    parser.add_argument(
        "--img-len",
        default=144,
        type=float,
        metavar="SX",
        help="length of the image, also the diameter of the FOV (unit: mm)",
    )
    parser.add_argument("--detr-num", default=500, type=int, metavar="NU", help="number of detector unit")
    parser.add_argument(
        "--detr-len", default=180, type=float, metavar="SU", help="length of the detector panel (unit: mm)"
    )
    parser.add_argument(
        "--lat-sampling", default=2, type=int, help="sample grid of X-ray in longitude direction"
    )
    parser.add_argument(
        "--sdd", default=1200, type=float, help="distance of the X-ray source to detector panel (unit: mm)"
    )
    parser.add_argument(
        "--sod", default=981, type=float, help="distance of the X-ray source to object axis (unit: mm)"
    )
    parser.add_argument("--off-u", default=0, type=float, help="detector rotation shift length (unit: mm)")
    parser.add_argument("--rotate-step", default=1.0, type=float, help="longitude step (deg) of ray")
    parser.add_argument(
        "--direction", default=-1, type=float, help="gantry rotating direction (clockwise / counter clockwise"
    )
    parser.add_argument(
        "--max-epoch", default=10, type=int, metavar="N", help="number of total epochs to run (default: 10)"
    )
    # gantry setting
    param = parser.parse_args()

    main(param)
