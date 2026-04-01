# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

"""
Static configuration for CT imaging geometry.

Inspired by ``RaysCfg`` in `torch-radon`_.

.. _torch-radon: https://github.com/matteo-ronchetti/torch-radon
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RaysCfg:
    """Immutable CT imaging system configuration.

    All fields are set at construction time and cannot be changed afterwards,
    mirroring the ``RaysCfg`` struct from
    `torch-radon <https://github.com/matteo-ronchetti/torch-radon>`_.

    Args:
        img_pixels: Number of image pixels along each axis.
        img_len: Diameter of the field of view (mm).
        detr_num: Number of detector elements.
        detr_len: Detector panel length (mm).
        lat_sampling: Lateral sampling grid multiplier.
        sdd: Source-to-detector distance (mm).
        sod: Source-to-object distance (mm).
        rotate_step: Rotation step in degrees.
    """

    img_pixels: int
    img_len: float
    detr_num: int
    detr_len: float
    lat_sampling: int
    sdd: float
    sod: float
    rotate_step: float
