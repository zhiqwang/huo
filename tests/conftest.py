# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

import types

import pytest
import torch


@pytest.fixture
def small_param():
    """Small CT geometry parameters for fast tests."""
    return types.SimpleNamespace(
        img_pixels=16,
        img_len=144.0,
        detr_num=20,
        detr_len=180.0,
        lat_sampling=2,
        sdd=1200.0,
        sod=981.0,
        rotate_step=30.0,
    )


@pytest.fixture
def medium_param():
    """Medium CT geometry parameters for integration tests."""
    return types.SimpleNamespace(
        img_pixels=32,
        img_len=144.0,
        detr_num=50,
        detr_len=180.0,
        lat_sampling=2,
        sdd=1200.0,
        sod=981.0,
        rotate_step=10.0,
    )


@pytest.fixture
def gantry_coords(small_param):
    """Compute gantry coordinates for the small_param fixture."""
    param = small_param
    img_step = param.img_len / param.img_pixels
    img_end = (param.img_len - img_step) / 2
    detr_step = param.detr_len / param.detr_num
    detr_end = (param.detr_len - detr_step) / 2

    gantry_view = torch.arange(0, 360, step=param.rotate_step)

    src_coor = (0, -param.sod)

    detr_coor_x = torch.linspace(-detr_end, detr_end, steps=param.detr_num)
    detr_coor_y = torch.zeros_like(detr_coor_x) + param.sdd - param.sod

    lat_end = param.img_len / 2
    lat_steps = param.lat_sampling * param.img_pixels + 1
    lat_grid = torch.linspace(-lat_end, lat_end, steps=lat_steps).unsqueeze(1)
    lat_grid -= src_coor[1]
    detr_coor_y -= src_coor[1]

    fand = (detr_coor_x**2 + detr_coor_y**2).sqrt()
    gantry_coor_x = lat_grid * detr_coor_x / fand
    gantry_coor_y = lat_grid * detr_coor_y / fand
    gantry_coor_y += src_coor[1]

    gantry_coor_x /= img_end
    gantry_coor_y /= img_end

    return types.SimpleNamespace(
        gantry_coor_x=gantry_coor_x,
        gantry_coor_y=gantry_coor_y,
        gantry_view=gantry_view,
        img_end=img_end,
        detr_end=detr_end,
    )


@pytest.fixture
def parallel_coords(small_param):
    """Compute parallel beam geometry coordinates for the small_param fixture.

    For parallel beam the gantry views cover [0, 180) and the
    sinogram is generated using the same ``scan`` helper as fan-beam
    but with parallel gantry coordinates (detector × lateral grid).
    """
    param = small_param
    img_step = param.img_len / param.img_pixels
    img_end = (param.img_len - img_step) / 2
    detr_step = param.detr_len / param.detr_num
    detr_end = (param.detr_len - detr_step) / 2

    # Parallel beam: half rotation
    gantry_view = torch.arange(0, 180, step=param.rotate_step)

    return types.SimpleNamespace(
        gantry_view=gantry_view,
        img_end=img_end,
        detr_end=detr_end,
    )
