---
tocdepth: 2
---

# Geometry

Python, TypeScript, the CLI, diagrams, and proofs use the conventions on this
page: **radians, pixel units, and `[angle, detector]` sinograms**. The public
parameter names and scanner orientation follow torch-radon's
[Python API](https://github.com/matteo-ronchetti/torch-radon/blob/5796b2a07e3bb0a248430e9a07fb48db5994d4ea/torch_radon/__init__.py)
and [fan-beam kernel](https://github.com/matteo-ronchetti/torch-radon/blob/5796b2a07e3bb0a248430e9a07fb48db5994d4ea/src/forward.cu).
The [Tutorial](tutorial.md) demonstrates the shared `RadonFanbeam` API.

## Fan-beam layout

Image columns increase along x and rows along y. At angle $\theta=0$, the
source is $S=(0,d_s)$ and detector coordinate $u$ is the point
$D_u=(u,-d_d)$. Thus the source is below the image centre in an array display.
The axial source-to-detector distance is $D=d_s+d_d$.

```{raw} html
:file: _static/figures/fanbeam-geometry.svg
```

**Zero-angle scanner geometry.** The open arrows mark distances; dashed
extensions identify their endpoints. The blue shape is an example object,
not a circular support constraint. Sizes and detector counts are schematic.
[Open the diagram](_static/figures/fanbeam-geometry.svg).

The table is the notation reference for the entire repository. Python and
TypeScript use identical field names, including snake case.

| Symbol | Field or quantity | Meaning |
| --- | --- | --- |
| $N$ | `resolution` | Pixels per image axis; image side length in pixel units. |
| $M$ | `det_count` | Number of detector elements. |
| $V$ | `len(angles)` / `angles.length` | Number of projection angles. |
| $\theta$ | `angle`, an entry of `angles` | Projection angle in radians. |
| $d_s$ | `source_distance` | Source to image centre, in pixels. |
| $d_d$ | `det_distance` | Image centre to detector plane, in pixels. |
| $D$ | `source_detector_distance` | Derived sum $d_s+d_d$. |
| $\Delta u$ | `det_spacing` | Detector-centre spacing, in pixels. |
| $u$ | Detector coordinate | Signed position along the detector. |
| $t$ | Ray distance | Distance from $S$ along a ray, in pixels. |
| $q_{\rm ray}$ | `ray_sampling` | Integration intervals per pixel along a ray. |
| $L=N$ | Ray-window length | Image side length, also used to normalize the iterative update. |

The gantry rotates by $R_{-\theta}$. To locate a pixel in the scanner frame,
backprojection uses the inverse rotation $R_\theta$:

$$
R_\theta=\begin{pmatrix}\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta\end{pmatrix},\qquad
\begin{pmatrix}x'\\y'\end{pmatrix}=R_\theta
\begin{pmatrix}x\\y\end{pmatrix}.
$$

With the downward y axis, a positive public angle rotates the gantry
counterclockwise on the page. All pixel centres being backprojected must be
in front of the source: $d_s-y'>0$. Intersecting the source-to-pixel line
with the detector plane gives

$$
u_\theta(p)=\frac{D x'}{d_s-y'}.
$$

The [Ray geometry proof](backward_propagation_proof.md#ray-geometry) derives
this formula and states what it establishes about residual backfill.

## Sampling grid

One image pixel has unit spacing. For row $r$, column $c$, and detector $i$,

$$
\begin{aligned}
x_c&=c-\frac{N-1}{2},& y_r&=r-\frac{N-1}{2},\\
u_i&=\left(i-\frac{M-1}{2}\right)\Delta u.
\end{aligned}
$$

Here $r,c=0,\ldots,N-1$ and $i=0,\ldots,M-1$. Image edges are at
$\pm N/2$, while the outermost pixel centres are at
$\pm a_x$, with $a_x=(N-1)/2$. Detector centres extend to
$\pm a_u$, with $a_u=(M-1)\Delta u/2$.

```{raw} html
:file: _static/figures/sampling-centres.svg
```

**Centres versus edges.** Both projection directions use these centres.
The implementation names the extents `pixel_extent` and `det_extent`.
[Open the diagram](_static/figures/sampling-centres.svg).

`FanbeamGeometry` owns these derived quantities. Both `RadonFanbeam`
implementations construct it once and cache the zero-angle ray coordinates.
Python's image sampler uses `grid_sample(..., align_corners=True)`;
TypeScript uses the same mapping from $\pm a_x$ to $\pm1$.
Backprojection constructs pixel centres directly with
`torch.meshgrid(..., indexing="ij")`. Both languages interpolate the detector
row at fractional index $u/\Delta u+(M-1)/2$, retaining its natural 1-D shape.

The fixed ray window has $K=q_{\rm ray}N$ intervals and $K+1$ samples:

$$
\begin{aligned}
\Delta t&=1/q_{\rm ray},& t_k&=d_s-N/2+k\Delta t,\\
x(t,u)&=\frac{tu}{\sqrt{u^2+D^2}},&
y(t,u)&=d_s-\frac{tD}{\sqrt{u^2+D^2}}.
\end{aligned}
$$

`compute_ray_coordinates` returns `ray_x` and `ray_y` normalized by
`pixel_extent`, with shape `[ray_sample_count, det_count]`. Python forms these
coordinates with outer products of ray distances and detector directions.
The code rotates these positions by $R_{-\theta}$ and sums interpolated values with weight
`ray_step`. Both endpoints receive the full weight. The window is not a
ray intersection with the full square, and can omit support near corners.

Image and detector interpolation use **zero-valued out-of-range neighbours**.
Half a cell beyond the last centre retains half the edge value; a full cell
beyond it contributes zero. No circular mask is applied. Backprojection
uniformly fills detector values along rays. These numerical rules define
Huo's operators; the proof explains why shared geometry alone does not make
them discrete transposes or inverses.

## Parameters

Both languages expose `RadonFanbeam` and an immutable `FanbeamGeometry`.
All geometry arguments are explicit except `ray_sampling`, which defaults
to `2`. Construct a new operator to change geometry or angles.

`resolution` and `det_count` must be integers at least two, `ray_sampling`
a positive integer, and scanner lengths positive and finite. The source
distance must exceed $N/2$ so the entire ray window is in front of the source.
Backprojection also checks $d_s-y'>0$ at every image pixel for each angle.

| Python / TypeScript field | CLI option | Unit |
| --- | --- | --- |
| `resolution` | `--resolution` | Pixel count |
| `source_distance` | `--source-distance` | Pixels |
| `det_distance` | `--det-distance` | Pixels |
| `det_count` | `--det-count` | Detector count |
| `det_spacing` | `--det-spacing` | Pixels |
| `ray_sampling` | `--ray-sampling` | Intervals per pixel |
| `angles` | `--angles` | NumPy angle file, radians |
| Uniform angle generation | `--angle-step` | Radians; generates $[0,2\pi)$ |

`--angles` preserves the given row order and supports nonuniform acquisitions.
It is mutually exclusive with `--angle-step`.

| Setting | Python CLI default | Browser demo |
| --- | --- | --- |
| `resolution` | `512` | `128` |
| `source_distance` | `3488` | `872` |
| `det_distance` | `778.6666666666666` | `194.66666666666666` |
| `det_count` | `500` | `200` |
| `det_spacing` | `1.28` | `0.8` |
| `ray_sampling` | `2` | `2` |
| Angular step | $\pi/180$ → `360` views | $\pi/90$ → `180` views |

To describe a physical scanner with pixel pitch $h$ mm/pixel, divide all
measured lengths by $h$ before constructing the operator. The example
source distance `872` is `981 mm / 1.125 mm per pixel`. If image values keep
the same numerical units, line integrals measured with a millimetre path
length must also be divided by $h$ to obtain pixel-length integrals.

### Migrating earlier Huo data

The earlier API used degrees, millimetres, a source on the negative y axis,
and `[detector, angle]` arrays. This release changes that interface. For the
same image orientation and acquisition, convert all four conventions together:

```python
pixel_size = old_volume_size / resolution
angles = np.pi - np.deg2rad(old_angles)
sinogram = np.ascontiguousarray(old_sinogram.T[:, ::-1] / pixel_size, dtype=np.float32)
source_distance = old_sod / pixel_size
det_distance = (old_sdd - old_sod) / pixel_size
det_spacing = (old_detector_length / det_count) / pixel_size
```

The detector reversal compensates for the new scanner orientation; a transpose
alone is insufficient. Keep the converted angle array with the data and pass
it through `angles` or `--angles`. `ray_sampling` replaces the old sampling
multiplier. The image itself is not reflected. The migration regression test
checks this conversion against the previous physical-coordinate sampler.
