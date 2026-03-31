# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

"""Command-line interface for CT reconstruction with ART."""

import argparse

import numpy as np
import torch

from huo.radon import RadonFanbeam


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="huo-art",
        description="CT image reconstruction using the Algebraic Reconstruction Technique (ART)",
    )
    parser.add_argument("sinogram", help="path to sinogram file (.npy)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="path to save the reconstructed image (.npy). If omitted the result is not saved.",
    )
    parser.add_argument(
        "--img-pixels", default=512, type=int, metavar="N", help="image pixels per axis (default: 512)"
    )
    parser.add_argument(
        "--img-len", default=144, type=float, metavar="MM", help="FOV diameter in mm (default: 144)"
    )
    parser.add_argument(
        "--detr-num", default=500, type=int, metavar="N", help="number of detector elements (default: 500)"
    )
    parser.add_argument(
        "--detr-len", default=180, type=float, metavar="MM", help="detector panel length in mm (default: 180)"
    )
    parser.add_argument(
        "--lat-sampling", default=2, type=int, metavar="N", help="lateral sampling multiplier (default: 2)"
    )
    parser.add_argument(
        "--sdd", default=1200, type=float, metavar="MM",
        help="source-to-detector distance in mm (default: 1200)",
    )
    parser.add_argument(
        "--sod", default=981, type=float, metavar="MM",
        help="source-to-object distance in mm (default: 981)",
    )
    parser.add_argument(
        "--rotate-step", default=1.0, type=float, metavar="DEG",
        help="rotation step in degrees (default: 1.0)",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Load sinogram
    sinogram = np.load(args.sinogram)
    print(f"sinogram: {sinogram.shape}")
    sinogram = torch.from_numpy(sinogram)

    # Build RadonFanbeam operator
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

    # Reconstruct
    img = radon.art(sinogram)
    print(f"image: {img.shape}")

    # Save output
    if args.output is not None:
        np.save(args.output, img.numpy())
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
