# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

import argparse

import numpy as np
import torch

from huo.radon import RadonFanbeam


def main(args):
    angles = torch.arange(0, 360, step=args.rotate_step)
    det_spacing = args.detr_len / args.detr_num

    radon = RadonFanbeam(
        resolution=args.img_pixels,
        angles=angles,
        source_distance=args.sod,
        det_distance=args.sdd - args.sod,
        det_count=args.detr_num,
        det_spacing=det_spacing,
        volume_size=args.img_len,
        lat_sampling=args.lat_sampling,
    )

    # Forward projection (uncomment to generate a sinogram from an image):
    # from PIL import Image
    # img = Image.open("./data/shepp2d.tif")
    # img = np.array(img) / 255.0
    # img = torch.from_numpy(img).type(torch.FloatTensor)
    # print(f"image: {img.shape}")
    # sinogram = radon.forward(img)
    # print(f"sinogram: {sinogram.shape}")

    # ART reconstruction
    sinogram = np.load("./data/sinogram.npy")
    print(f"sinogram: {sinogram.shape}")
    sinogram = torch.from_numpy(sinogram)

    img = radon.art(sinogram)
    print(f"image: {img.shape}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Algebraic reconstruction technique")

    parser.add_argument("--img-pixels", default=512, type=int, metavar="NX", help="sample pixels of image")
    parser.add_argument(
        "--img-len",
        default=144,
        type=float,
        metavar="SX",
        help="diameter of the FOV (unit: mm)",
    )
    parser.add_argument("--detr-num", default=500, type=int, metavar="NU", help="number of detector elements")
    parser.add_argument(
        "--detr-len", default=180, type=float, metavar="SU", help="detector panel length (unit: mm)"
    )
    parser.add_argument("--lat-sampling", default=2, type=int, help="lateral sampling grid multiplier")
    parser.add_argument(
        "--sdd", default=1200, type=float, help="source-to-detector distance (unit: mm)"
    )
    parser.add_argument(
        "--sod", default=981, type=float, help="source-to-object distance (unit: mm)"
    )
    parser.add_argument("--rotate-step", default=1.0, type=float, help="rotation step (degrees)")

    args = parser.parse_args()
    main(args)
