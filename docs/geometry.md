---
tocdepth: 2
---

# Geometry

Python, TypeScript, the CLI, diagrams, and proofs use the conventions on this
page: **radians, pixel units, and `[angle, detector]` sinograms**. The public
parameter names and scanner orientation follow torch-radon's
[Python API](https://github.com/matteo-ronchetti/torch-radon/blob/5796b2a07e3bb0a248430e9a07fb48db5994d4ea/torch_radon/__init__.py)
and [fan-beam kernel](https://github.com/matteo-ronchetti/torch-radon/blob/5796b2a07e3bb0a248430e9a07fb48db5994d4ea/src/forward.cu).
The [Tutorial](tutorial.md) demonstrates Python's `FanbeamProjector` and
loss-based reconstruction. The browser jax-js backend shares this geometry
and differentiates the same sampled forward model.

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
| $L=N$ | Ray-window length | Image side length; historical ART used it to normalize residual updates. |

The gantry rotates by $R_{-\theta}$. To locate a pixel in the scanner frame,
the historical residual-backfill model uses the inverse rotation $R_\theta$:

$$
R_\theta=\begin{pmatrix}\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta\end{pmatrix},\qquad
\begin{pmatrix}x'\\y'\end{pmatrix}=R_\theta
\begin{pmatrix}x\\y\end{pmatrix}.
$$

With the downward y axis, a positive public angle rotates the gantry
counterclockwise on the page. In that backfill model, pixel centres being backprojected must be
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

`FanbeamGeometry` owns these derived quantities. Python's `FanbeamProjector`
implementations and TypeScript's `FanbeamProjector` cache zero-angle ray coordinates.
PyTorch uses `grid_sample(..., align_corners=True)`; JAX uses linear
`map_coordinates` with partial zero padding and pixel-index coordinates.
TypeScript uses native jax-js gathers with the same mapping from
$\pm a_x$ to $\pm1$ and partial zero padding. All current backends differentiate
their actual image sampling weights to propagate projection-loss gradients.
The historical residual backfill interpolated detector rows at
$u/\Delta u+(M-1)/2$, which is a different operation.

The fixed ray window has $K=q_{\rm ray}N$ intervals and $K+1$ samples:

$$
\begin{aligned}
\Delta t&=1/q_{\rm ray},& t_k&=d_s-N/2+k\Delta t,\\
x(t,u)&=\frac{tu}{\sqrt{u^2+D^2}},&
y(t,u)&=d_s-\frac{tD}{\sqrt{u^2+D^2}}.
\end{aligned}
$$

`compute_ray_coordinates` returns `ray_x` and `ray_y` normalized by
`pixel_extent`, with shape `[ray_sample_count, det_count]`. The shared NumPy
implementation forms these coordinates in float64 with outer products, then
converts to the requested float32/float64 dtype. Each backend prepares its own
arrays from those coordinates outside differentiation.
The code rotates these positions by $R_{-\theta}$ and sums interpolated values with weight
`ray_step`. Both endpoints receive the full weight. The window is not a
ray intersection with the full square, and can omit support near corners.

Image and detector interpolation use **zero-valued out-of-range neighbours**.
Half a cell beyond the last centre retains half the edge value; a full cell
beyond it contributes zero. No circular mask is applied. Browser backprojection
uniformly fills detector values along rays. Its weights differ from the
transpose of image sampling; shared geometry alone does not identify those maps.
The Python autograd path uses the sampled transpose internally.

## Parameters

Python constructs `FanbeamGeometry` explicitly, then passes it and the angles
to `huo.torch.FanbeamProjector(...)` or `huo.jax.FanbeamProjector.create(...)`.
TypeScript uses `new FanbeamProjector(geometry, angles)` with the same geometry
fields. The browser helper is asynchronous to support rendering and cancellation;
its loss and Adam update semantics match Python.

`resolution` and `det_count` must be integers at least two, `ray_sampling`
a positive integer, and scanner lengths positive and finite. The source
distance must exceed $N/2$ so the entire ray window is in front of the source.
The historical backfill additionally required $d_s-y'>0$ at each backprojected pixel.

| Geometry field | Python CLI | Unit |
| --- | --- | --- |
| `resolution` | `--resolution`, required | Pixel count |
| `source_distance` | `--source-distance`, required | Pixels |
| `det_distance` | `--det-distance`, required | Pixels |
| `det_count` | Inferred from the measurement's final dimension | Detector count |
| `det_spacing` | `--det-spacing`, required | Pixels |
| `ray_sampling` | `--ray-sampling`, default `2` | Intervals per pixel |
| `angles` | `--angles`, required | NumPy angle file in radians |

The angle sequence preserves measurement row order and may be nonuniform.
The Python CLI does not infer acquisition angles from a default step.

## Physical-unit measurements

For pixel pitch $h$ mm/pixel, divide scanner lengths by $h$ before constructing
the geometry. For example, source distance `872` corresponds to
`981 mm / 1.125 mm per pixel`. If image values retain the same numerical units,
line integrals measured using millimetre path lengths must also be divided by
$h$ to obtain pixel-length integrals.

For data from a scanner convention with the source on negative y, positive
$R_\theta$ ray rotation, degrees, and detector-major arrays, one corresponding
conversion is:

```python
angles = np.pi - np.deg2rad(acquisition_angles)
measured = np.ascontiguousarray(detector_major_data.T[:, ::-1] / pixel_pitch,
                                dtype=np.float32)
source_distance = source_to_centre_mm / pixel_pitch
det_distance = (source_to_detector_mm - source_to_centre_mm) / pixel_pitch
det_spacing = detector_pitch_mm / pixel_pitch
```

The reversal and angle conversion depend on that stated scanner convention.
Keep the converted angle array with the measurements. The physical-coordinate
regression test verifies the conversion against an independent sampler.
