# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

"""Reconstruct NumPy measurements by minimizing projection loss with Adam."""

import argparse

import numpy as np
import torch

from huo.geometry import FanbeamGeometry
from huo.projector import FanbeamProjector
from huo.reconstruction import reconstruct


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="huo-reconstruct", description="Optimize a fan-beam image against measured line integrals"
    )
    parser.add_argument("sinogram", help="NumPy measurements with shape [..., angle, detector]")
    parser.add_argument(
        "--angles", required=True, help="NumPy 1-D angles in radians, in measurement row order"
    )
    parser.add_argument("-o", "--output", required=True, help="output NumPy image file")
    parser.add_argument("--resolution", required=True, type=int, help="pixels per image axis")
    parser.add_argument(
        "--source-distance", required=True, type=float, help="source to image centre, in pixels"
    )
    parser.add_argument(
        "--det-distance", required=True, type=float, help="image centre to detector plane, in pixels"
    )
    parser.add_argument("--det-spacing", required=True, type=float, help="detector-centre spacing, in pixels")
    parser.add_argument(
        "--ray-sampling", default=2, type=int, help="integration intervals per pixel (default: 2)"
    )
    parser.add_argument("--steps", default=300, type=int, help="Adam iterations (default: 300)")
    parser.add_argument("--lr", default=1e-2, type=float, help="Adam learning rate (default: 0.01)")
    parser.add_argument("--allow-negative", action="store_true", help="disable nonnegative image projection")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        values = np.asarray(np.load(args.sinogram), dtype=np.float32)
        if values.ndim < 2:
            raise ValueError("sinogram must have shape [..., angle, detector]")
        geometry = FanbeamGeometry(
            args.resolution,
            args.source_distance,
            args.det_distance,
            values.shape[-1],
            args.det_spacing,
            args.ray_sampling,
        )
        projector = FanbeamProjector(geometry, np.load(args.angles))
        measured = torch.from_numpy(values)
        image = reconstruct(
            projector, measured, steps=args.steps, lr=args.lr, nonnegative=not args.allow_negative
        )
        with torch.no_grad():
            initial_loss = 0.5 * measured.square().mean()
            final_loss = 0.5 * (projector(image) - measured).square().mean()
        np.save(args.output, image.numpy())
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"measurements: {tuple(measured.shape)} [..., angle, detector]")
    print(f"image: {tuple(image.shape)}; steps: {args.steps}")
    print(f"loss: {initial_loss.item():.6g} -> {final_loss.item():.6g}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
