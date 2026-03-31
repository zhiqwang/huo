# Huo ART Tutorial

This tutorial walks through the **Algebraic Reconstruction Technique (ART)**
as implemented in this repository. It covers the algorithm, the CT geometry
setup, and how to use both the Python library and the browser-based demo.

## Table of Contents

- [Background](#background)
- [Algorithm Overview](#algorithm-overview)
- [CT Geometry Setup](#ct-geometry-setup)
- [Python Usage](#python-usage)
  - [Coordinate Preparation](#coordinate-preparation)
  - [Forward Projection (Scan)](#forward-projection-scan)
  - [ART Reconstruction](#art-reconstruction)
  - [CLI Tool](#cli-tool)
- [Browser Demo (TypeScript)](#browser-demo-typescript)
- [Parameter Reference](#parameter-reference)

---

## Background

In Computed Tomography (CT), an X-ray source rotates around an object and a
detector records the intensity of the X-rays that pass through. The collection
of all these measurements at every angle is called a **sinogram**.

Reconstructing the original image from its sinogram is the central problem of
CT imaging. The **Algebraic Reconstruction Technique (ART)** is one of the
earliest iterative methods for solving this problem.

## Algorithm Overview

ART reconstructs the image one projection angle at a time. Given a measured
sinogram **S** and an initial image estimate **I** (typically all zeros), the
algorithm repeats the following for each angle θ in random order:

```
1. predicted  ← forward_project(I, θ)      # what the detector would see
2. residual   ← (S[θ] − predicted) / L      # error, scaled by image length L
3. correction ← back_project(residual, θ)   # spread the error back into image space
4. I          ← clamp(I + correction, 0, ∞) # update and enforce non-negativity
```

This is the **Kaczmarz method** applied to the CT linear system. Visiting
angles in random order improves convergence.

### Forward Projection

Forward projection computes the line integral of the current image estimate
along each X-ray path for a given angle. This repository implements it by:

1. Rotating the fan-beam gantry coordinates by the projection angle.
2. Sampling the image at these rotated coordinates using **bilinear
   interpolation** (equivalent to `F.grid_sample` in PyTorch).
3. Summing the samples along the lateral (ray) direction to produce one
   sinogram column.

### Back-Projection

Back-projection distributes a sinogram column back into image space:

1. For each pixel in the image grid, compute which detector element it maps to
   under the current projection angle using the fan-beam geometry.
2. Interpolate the sinogram value at that detector position.
3. Place the interpolated value at the corresponding pixel.

This is implemented using `F.affine_grid` + `F.grid_sample` in PyTorch, and
direct computation in the TypeScript port.

## CT Geometry Setup

Both implementations use a **fan-beam** geometry with the following coordinate
conventions (matching the `tools/projection.py` defaults):

```
                  X-ray Source
                  (0, −SOD)
                      │
                      │  SOD = 981 mm
                      │
              ────────┼────────  Object / FOV (diameter = 144 mm)
                      │
                      │  SDD − SOD = 219 mm
                      │
              ════════╪════════  Detector panel (180 mm, 500 elements)
```

- The **source** sits at `(0, −SOD)` in the un-rotated frame.
- The **detector** is a flat panel at `y = SDD − SOD` from the object centre.
- At each angle the entire gantry (source + detector) rotates around the
  object centre.

### Coordinate Preparation

The ``RadonFanbeam`` class pre-computes all gantry coordinates internally.
You only need to specify the geometry once:

```python
import torch
from huo import RadonFanbeam

angles = torch.arange(0, 360, step=1.0)

radon = RadonFanbeam(
    resolution=512,          # image pixels per axis
    angles=angles,           # projection angles in degrees
    source_distance=981,     # SOD — source to rotation centre (mm)
    det_distance=219,        # rotation centre to detector (mm)
    det_count=500,           # number of detector elements
    det_spacing=0.36,        # detector element spacing (mm)
    volume_size=144,         # FOV diameter (mm)
    lat_sampling=2,          # lateral sampling multiplier
)
```

Under the hood the constructor runs the same coordinate computation
that was previously in ``tools/projection.py``:

```python
# Derived quantities (computed automatically)
img_step  = volume_size / resolution
img_end   = (volume_size - img_step) / 2
det_step  = det_length / det_count
det_end   = (det_length - det_step) / 2

# Projection angles
gantry_view = torch.arange(0, 360, step=rotate_step)

# Source position (top of gantry)
src_coor = (0, -sod)

# Detector element positions
detr_coor_x = torch.linspace(-detr_end, detr_end, steps=detr_num)
detr_coor_y = torch.zeros_like(detr_coor_x) + sdd - sod

# Lateral sampling grid (along each ray)
lat_end   = img_len / 2
lat_steps = lat_sampling * img_pixels + 1
lat_grid  = torch.linspace(-lat_end, lat_end, steps=lat_steps).unsqueeze(1)

# Shift origin to X-ray source
lat_grid    -= src_coor[1]
detr_coor_y -= src_coor[1]

# Fan-beam distances
fand = (detr_coor_x ** 2 + detr_coor_y ** 2).sqrt()

# Gantry coordinates  [lat_steps, detr_num]
gantry_coor_x = lat_grid * detr_coor_x / fand
gantry_coor_y = lat_grid * detr_coor_y / fand + src_coor[1]

# Normalise to [-1, 1] for grid_sample
gantry_coor_x /= img_end
gantry_coor_y /= img_end
```

## Python Usage

### Forward Projection (Scan)

Use ``forward()`` to compute a full sinogram from an image:

```python
sinogram = radon.forward(img)
# sinogram shape: [det_count, num_angles]
```

Where ``img`` is a ``torch.Tensor`` of shape ``[resolution, resolution]``.

### ART Reconstruction

Reconstruct an image from a sinogram:

```python
reconstructed = radon.art(sinogram)
# reconstructed shape: [resolution, resolution]
```

### CLI Tool

The `tools/projection.py` script wraps the full pipeline:

```bash
# Reconstruct from an existing sinogram
python tools/projection.py \
    --img-pixels 512 \
    --img-len 144 \
    --detr-num 500 \
    --detr-len 180 \
    --lat-sampling 2 \
    --sdd 1200 \
    --sod 981 \
    --rotate-step 1.0
```

The script expects `./data/sinogram.npy`. To generate a sinogram from an image
instead, uncomment the forward-projection block in `tools/projection.py`
(lines 48–54):

```python
# Uncomment these lines for forward projection:
# img = Image.open('./data/shepp2d.tif')
# img = np.array(img) / 255.
# img = torch.from_numpy(img).type(torch.FloatTensor)
# sinogram = scan(img, gantry_coor_x, gantry_coor_y, gantry_view, param)
```

## Browser Demo (TypeScript)

The TypeScript implementation in `js/` provides an interactive, in-browser
visualisation of both forward projection and ART reconstruction.

### Running the Demo

```bash
cd js
npm install
npm run build
npx serve .
```

Open `http://localhost:3000/demo/` in a modern browser.

### What the Demo Shows

1. **Phase 1 — Forward Projection:** A Shepp-Logan-style phantom is generated
   and forward-projected angle by angle. The sinogram canvas updates live so
   you can watch it being built column by column.

2. **Phase 2 — ART Reconstruction:** The sinogram is fed to the ART algorithm.
   The reconstruction canvas updates after every few angle iterations,
   showing the image gradually emerging from noise.

The **Frame delay** slider (0–200 ms) controls the animation speed, allowing
you to slow down or speed up the visualisation.

### TypeScript API

The core functions in `js/src/art.ts` mirror the Python API:

```typescript
import { forward, backprojection, scan, art, type CTParam } from "./art.js";

const param: CTParam = {
  imgPixels: 128,
  imgLen: 144,
  detrNum: 200,
  detrLen: 180,
  latSampling: 2,
  sdd: 1200,
  sod: 981,
  rotateStep: 2,
};

// Forward projection for one angle
const sinoCol = await forward(img, gantryCoordX, gantryCoordY, angle, param);

// Back-projection for one angle
const bpImg = await backprojection(sinoData, imgEnd, detEnd, angle, param);

// Full forward projection → sinogram
const sinogram = await scan(img, gantryCoordX, gantryCoordY, angles, param);

// ART reconstruction → image
const reconstructed = await art(
  sinogram, imgEnd, detEnd, gantryCoordX, gantryCoordY, angles, param,
);
```

Both `scan()` and `art()` accept optional `onProgress` callbacks and a `delay`
parameter for animation purposes.

## Parameter Reference

| Parameter | Python CLI flag | TypeScript field | Default | Unit | Description |
|---|---|---|---|---|---|
| Image pixels | `--img-pixels` | `imgPixels` | 512 | px | Pixels along each axis of the reconstruction grid |
| Image length | `--img-len` | `imgLen` | 144 | mm | Diameter of the field of view (FOV) |
| Detector count | `--detr-num` | `detrNum` | 500 | — | Number of detector elements |
| Detector length | `--detr-len` | `detrLen` | 180 | mm | Physical length of the detector panel |
| Lateral sampling | `--lat-sampling` | `latSampling` | 2 | — | Multiplier for the number of sample points along each ray |
| SDD | `--sdd` | `sdd` | 1200 | mm | Source-to-detector distance |
| SOD | `--sod` | `sod` | 981 | mm | Source-to-object (rotation centre) distance |
| Rotation step | `--rotate-step` | `rotateStep` | 1.0 | deg | Angular step between successive projections |
| Max epochs | `--max-epoch` | — | 10 | — | Number of full ART passes (Python CLI only) |
