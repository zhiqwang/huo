# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

"""Command-line interface using the same geometry names and units as the API."""

import argparse
import math

import numpy as np
import torch

from huo.radon import RadonFanbeam


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="huo-art",
        description="Fan-beam residual reconstruction; angles in radians, distances in pixels",
    )
    parser.add_argument("sinogram", help="NumPy sinogram with shape [angle, detector]")
    parser.add_argument("-o", "--output", help="save the reconstructed image as a NumPy array")
    parser.add_argument("--resolution", default=512, type=int, help="pixels per image axis (default: 512)")
    parser.add_argument("--source-distance", default=3488.0, type=float, help="source to centre, in pixels")
    parser.add_argument(
        "--det-distance", default=778.6666666666666, type=float, help="centre to detector plane, in pixels"
    )
    parser.add_argument("--det-count", default=500, type=int, help="number of detector elements")
    parser.add_argument("--det-spacing", default=1.28, type=float, help="detector spacing, in pixels")
    parser.add_argument("--ray-sampling", default=2, type=int, help="integration intervals per pixel")
    angle_group = parser.add_mutually_exclusive_group()
    angle_group.add_argument(
        "--angle-step",
        default=math.pi / 180,
        type=float,
        help="uniform angular step in radians (default: pi/180)",
    )
    angle_group.add_argument("--angles", help="NumPy 1-D angle array in radians, in sinogram row order")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.angles is not None:
        angles = np.load(args.angles)
    else:
        if not math.isfinite(args.angle_step) or args.angle_step <= 0:
            parser.error("angle-step must be positive and finite (in radians)")
        # Form the angle count in double precision, then round values to float32.
        angles = np.arange(0, 2 * math.pi, args.angle_step, dtype=np.float64)
    radon = RadonFanbeam(
        resolution=args.resolution,
        angles=angles,
        source_distance=args.source_distance,
        det_distance=args.det_distance,
        det_count=args.det_count,
        det_spacing=args.det_spacing,
        ray_sampling=args.ray_sampling,
    )
    sinogram = torch.from_numpy(np.asarray(np.load(args.sinogram), dtype=np.float32))
    image = radon.art(sinogram)
    print(f"sinogram: {tuple(sinogram.shape)} [angle, detector]")
    print(f"image: {tuple(image.shape)}")
    if args.output is not None:
        np.save(args.output, image.numpy())
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
