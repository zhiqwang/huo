# Copyright (c) 2022, Zhiqiang Wang. All rights reserved.

import numpy as np
import torch
import torch.nn.functional as F


def fbp(sinogram, img_end, detr_end, gantry_view, param):
    """
    Filtered back-projection for parallel beam geometry.

    Applies the ramp (Ram-Lak) filter to each projection, then
    back-projects the filtered data using parallel beam geometry.

    Reference: https://astra-toolbox.com/docs/geom2d.html#parallel

    Args:
        sinogram: Sinogram tensor [detr_num, num_angles].
        img_end: Image coordinate boundary (half the FOV extent).
        detr_end: Detector coordinate boundary (half the panel extent).
        gantry_view: Projection angles in degrees (typically [0, 180)).
        param: CT geometry parameters.

    Returns:
        Reconstructed image [img_pixels, img_pixels].
    """
    # Apply ramp filter
    filtered = ramp_filter(sinogram, param)
    filtered = filtered.unsqueeze(0).unsqueeze(0)

    angle_step_rad = param.rotate_step * np.pi / 180.0
    img = torch.zeros(param.img_pixels, param.img_pixels)

    for i, view in enumerate(gantry_view):
        sinogram_col = filtered[:, :, :, i]
        img += parallel_backprojection(sinogram_col, img_end, detr_end, view.item(), param)

    img *= angle_step_rad
    return img


def ramp_filter(sinogram, param):
    """
    Apply ramp (Ram-Lak) filter to each projection in the sinogram.

    Pads each column to the next power of two, multiplies in the
    frequency domain by |freq|, and transforms back.

    Args:
        sinogram: Sinogram tensor [detr_num, num_angles].
        param: CT geometry parameters.

    Returns:
        Filtered sinogram [detr_num, num_angles].
    """
    detr_step = param.detr_len / param.detr_num
    num_angles = sinogram.shape[1]

    # Pad to next power of 2 for efficient FFT
    pad_len = 1
    while pad_len < 2 * param.detr_num:
        pad_len *= 2

    # Ramp filter: |frequency|
    freqs = torch.fft.rfftfreq(pad_len, d=detr_step)
    ramp = torch.abs(freqs)

    filtered = torch.zeros_like(sinogram)
    for i in range(num_angles):
        padded = torch.zeros(pad_len)
        padded[: param.detr_num] = sinogram[:, i]
        f = torch.fft.rfft(padded)
        f = f * ramp
        filtered[:, i] = torch.fft.irfft(f, n=pad_len)[: param.detr_num]

    return filtered


def parallel_backprojection(sinogram, img_end, detr_end, view, param):
    """
    Parallel beam back-projection for a single angle.

    For each pixel at position (x, y), the detector coordinate is
    t = x * cos(theta) + y * sin(theta), scaled from image space to
    detector space.

    Args:
        sinogram: Sinogram column [1, 1, detr_num].
        img_end: Image coordinate boundary.
        detr_end: Detector coordinate boundary.
        view: Projection angle in degrees.
        param: CT geometry parameters.

    Returns:
        Back-projected image [img_pixels, img_pixels].
    """
    sinogram = sinogram.unsqueeze(3)
    view = torch.tensor([view * np.pi / 180.0])

    theta = torch.tensor(
        [[torch.cos(view), torch.sin(view), 0], [-torch.sin(view), torch.cos(view), 0]]
    ).unsqueeze(0)

    img_grid_rot = F.affine_grid(theta, (1, 1, param.img_pixels, param.img_pixels)).squeeze(0)

    # Parallel beam: detector position = rotated x scaled to detector space
    samples = img_grid_rot[:, :, 0] * img_end / detr_end

    samples = samples.reshape(-1, 1)
    samples = samples.unsqueeze(2).unsqueeze(0)
    samples = torch.cat([torch.zeros_like(samples), samples], 3)

    img = F.grid_sample(sinogram, samples, align_corners=True)
    img = img.squeeze(0).squeeze(0)
    img = img.reshape(param.img_pixels, param.img_pixels)

    return img
