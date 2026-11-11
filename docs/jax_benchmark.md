# JAX Reproduction and CPU Comparison

This experiment reproduces Huo's sampled forward model, automatic derivatives,
and Adam reconstruction in JAX. It measures whether changing the differentiation
framework helps this particular implementation. These measurements were made
with the prototype in `benchmarks/jax_projector.py`, before public integration.
The native sampling core now lives in `huo.jax` and is shared with that benchmark;
its custom-JVP variant remains experimental. See the [JAX guide](jax.md) for
the supported interface and the [API reference](api.md) for both backends.

## Implementations

All variants use the same fixed normalized ray coordinates, angle order,
bilinear interpolation, partial zero padding, and ray-step normalization.
JAX's `map_coordinates(order=1, mode="constant", cval=0)` receives pixel-index
coordinates converted from Huo's normalized grid. Geometry arrays are explicit
runtime arguments to the compiled functions, rather than closed-over constants
whose interpolation weights could all be evaluated during compilation.

| Backend | Angle traversal and differentiation |
| --- | --- |
| PyTorch | Current `FanbeamProjector`, private transpose Function, and native sampling gradient. |
| JAX vmap | Vectorize over angles and images; native automatic differentiation. |
| JAX scan | Iterate over angles with `lax.scan`; checkpoint each view to recompute sampling intermediates during backward. |
| JAX custom JVP | The same scan/checkpoint implementation, with the explicit linear rule $\dot y=A\dot x$. JAX derives its transpose. |

None of the JAX implementations contains a hand-written backprojection or VJP.
The native variants do not require a custom derivative rule at all. All scanner
arrays are treated as constants when differentiating.

Reconstruction minimizes $\frac12\operatorname{mean}((Ax-y)^2)$ from zeros,
using Adam with learning rate `0.03`, betas `(0.9, 0.999)`, epsilon `1e-8`,
and a nonnegative clamp after each update. JAX uses Optax and compiles the
whole 30-step loop. PyTorch uses the existing `reconstruct` helper. Both
initialize optimizer state for each solve and return only the final image.
Compiling the complete JAX loop permits optimizations across its iterations;
its performance need not equal 30 separately dispatched gradient calls.

## Correctness

The prototype regression tests cover:

- float32/float64 forward results and VJPs against PyTorch, with odd/even image
  sizes, unsorted/repeated angles, and multiple leading batch dimensions;
- JVPs, Jacobians, and Hessians against a matrix assembled from PyTorch basis
  projections, plus first- and second-order finite differences;
- the shared partial-zero-padding fixture;
- 40 Adam steps compared with PyTorch, for both nonnegative and signed images.

The integrated backend additionally tests public projection, gradients, and
reconstruction against PyTorch, PyTree/JIT behavior, validation, CLI operation,
and installations without PyTorch. These integration tests do not turn the
prototype timing data below into measurements of the public API.

Every benchmark run also compares outputs with PyTorch on that exact workload.
Relative L2 error must be below `1e-4` for operators/derivatives and `1e-3` for
the final reconstructed image; nonfinite results fail the run. Reconstruction
loss is evaluated with the PyTorch projector for every backend, and must
decrease. Actual errors and losses are retained in the raw measurements.

## Measurement method

Python 3.12, the locked CPU dependencies, and Linux are used. Each
backend/workload/operation runs in a fresh process pinned to the same logical
CPU. PyTorch has one intra-op and one inter-op thread. JAX is restricted to
the same CPU affinity, and the persistent compilation cache is disabled.

Inputs and measurements are identical float32 arrays. JAX results are blocked
until ready after every timed call. Compilation is measured using
`jit(...).lower(...).compile()`; the first execution and warm medians are
recorded separately. Forward/gradient timings use seven samples, while full
reconstruction uses three. Two warmup calls precede sampling. Framework import,
device initialization, and input preparation are recorded separately from
operator execution. The machine is shared, so these are local measurements,
not portable performance guarantees.

PyTorch timings include its public input validation. The JAX prototype assumes
valid input tensors; it does not reproduce the complete module/state/validation
API. No GPU, `torch.compile`, or specialized CT kernel is included in this
comparison.

| Workload | Image shape | Angles | Detectors |
| --- | --- | --- | --- |
| small | `[1, 32, 32]` | 24 | 40 |
| medium | `[1, 128, 128]` | 48 | 160 |
| large | `[1, 256, 256]` | 90 | 320 |
| batch | `[4, 128, 128]` | 48 | 160 |

Source distance is `4*N`, detector distance is `2*N`, detector spacing is
`1.3`, and ray sampling is `2` for every workload. JVP/HVP measurements use
the medium workload with the same random direction supplied as a runtime
argument. JVP timing includes both primal output and tangent; HVP timing
includes both gradient and Hessian-vector product, using forward-over-reverse
AD without assembling a Hessian. Neither benchmark specializes to a constant
all-ones direction or times only a derivative after discarding its primal.

## Measured results

Recorded on **2026-11-12 (Asia/Shanghai)** on an Intel Xeon Gold 6336Y,
pinned to logical CPU 2. Versions: Python 3.12.13, PyTorch 2.15.0+cpu,
JAX/JAXlib 0.12.0.1, Optax 0.2.8, NumPy 2.5.4.

All timings below are warm medians in **milliseconds**. Each column names
the actual implementation; the JAX variants are not combined into a single
best-case number.


### Forward projection

| Workload | PyTorch | JAX vmap | JAX scan | JAX custom JVP |
| --- | --- | --- | --- | --- |
| small | 1.56 | 0.72 | 0.30 | 0.29 |
| medium | 21.70 | 78.50 | 21.51 | 21.38 |
| large | 179.12 | 561.62 | 194.16 | 191.22 |
| batch | 60.21 | 203.26 | 114.91 | 115.04 |

### Loss plus image gradient

| Workload | PyTorch | JAX vmap | JAX scan | JAX custom JVP |
| --- | --- | --- | --- | --- |
| small | 3.77 | 2.47 | 1.14 | 1.14 |
| medium | 52.47 | 151.14 | 52.98 | 53.65 |
| large | 434.28 | 1196.29 | 448.61 | 442.28 |
| batch | 137.02 | 404.73 | 186.46 | 212.16 |

### Complete 30-step Adam reconstruction

| Workload | PyTorch | JAX vmap | JAX scan | JAX custom JVP |
| --- | --- | --- | --- | --- |
| small | 117.84 | 25.63 | 33.40 | 33.42 |
| medium | 1596.96 | 1067.01 | 1611.36 | 1595.33 |
| large | 13475.10 | 8607.48 | 13308.29 | 13294.44 |
| batch | 4084.30 | 4528.13 | 5573.37 | 6393.13 |

The native scan loss/gradient is **3.29x faster** on the small workload,
approximately equal to PyTorch on medium and large, and **1.36x slower**
on the batched workload. Vectorizing all views makes standalone loss/gradient
slower than PyTorch on medium, large, and batch.

Compiling the entire vmap-based solver changes the tradeoff: 30-step solves
are **4.60x**, **1.50x**, and **1.57x faster** on small, medium, and large,
respectively. The batched solve is **1.11x slower**. This is an optimization
of the complete program, not evidence that the individual interpolation or
transpose kernel is faster. The explicit custom JVP has no consistent
advantage over native scan autodiff and is slower on the batched workload.


### JVP and HVP

| Operation (medium workload) | PyTorch | JAX vmap | JAX scan | JAX custom JVP |
| --- | --- | --- | --- | --- |
| Primal + JVP | 47.07 | 95.07 | 39.58 | 42.65 |
| Gradient + HVP | 111.72 | 216.45 | 92.16 | 92.80 |

Native scan gives **1.19x** and **1.21x** speedups for these JVP/HVP
programs. Adding the explicit JVP rule does not improve those results.

### Compilation and first use

| Workload | PyTorch first solve (ms) | JAX vmap compile (ms) | JAX vmap compile + first solve (ms) |
| --- | --- | --- | --- |
| small | 951.29 | 620.89 | 650.50 |
| medium | 2451.53 | 546.64 | 1654.99 |
| large | 14315.38 | 552.32 | 9462.18 |
| batch | 4932.56 | 562.31 | 5211.44 |

First-use numbers exclude the separately recorded framework/input setup.
PyTorch's first optimizer call also incurs lazy initialization, so its first
solve is slower than its warm solve. JAX compilation is paid for each distinct
compiled signature; the warm results assume reuse. Changing shapes or creating
fresh compiled closures can change that cost. The raw JSON records compile and
first-execution timings for every variant, not only vmap.


### Memory

| Backend | Large loss/gradient peak RSS (MiB) | Large reconstruction peak RSS (MiB) | XLA loss/gradient temporary buffers (MiB) |
| --- | --- | --- | --- |
| PyTorch | 300.6 | 378.1 | — |
| JAX vmap | 1475.6 | 1272.7 | 1183.56 |
| JAX scan | 285.6 | 315.9 | 8.37 |
| JAX custom JVP | 277.1 | 323.1 | 8.37 |

RSS is the Linux `/proc/self/status` **VmHWM** for the worker's address
space, including runtime imports, compilation, executable state, allocator
pools, and tensor buffers. It is not tensor-only memory. This avoids the
pre-exec parent high-water mark that can contaminate `ru_maxrss`.

XLA temporary-buffer estimates describe a compiled executable and are a
separate metric; they must not be compared directly with PyTorch saved-tensor
bytes or process RSS. The scan variants avoid the large all-view temporary
arrays. The faster large vmap reconstruction uses **3.37x the process peak
RSS** of the PyTorch solve. JAX does not provide an automatic memory win.


### Numerical agreement in these workloads

| Output | Maximum relative L2 error versus PyTorch |
| --- | --- |
| Projection | 5.424e-06 |
| Loss / image gradient | 5.778e-07 |
| Primal / JVP | 2.628e-06 |
| Gradient / HVP | 1.773e-06 |
| Reconstructed image | 7.560e-07 |

Every reconstructed image reduces the measured projection loss. For example,
the large workload falls from approximately `681.435` to `34.0306` after 30
steps for all four backends. These are parity and loss-reduction checks,
not claims of a converged or unique CT reconstruction.

### Implication for Huo

A pure JAX implementation substantially simplifies the derivative machinery:
native differentiation already reproduces the sampled transpose and higher
image derivatives. A custom JVP is optional here.

These CPU results support offering JAX as an optional backend for small
problems, complete compiled solvers, and derivative-heavy experiments. They
do **not** support replacing the current PyTorch implementation solely for
speed: medium/large individual steps gain little, the tested batch is slower,
and full-view vectorization can consume much more memory. GPU performance,
CPU scaling across more cores, and compiled PyTorch remain separate experiments.


## Reproduce

```bash
uv sync --locked --python 3.12 --all-extras --group test
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync pytest tests/test_jax_reference.py -q
uv run --no-sync python -m benchmarks.compare_jax --cpu 2 \
  --output benchmarks/results/jax_cpu.json
uv run --no-sync python -m benchmarks.compare_jax --cpu 2 \
  --cases medium --operations jvp hvp \
  --output benchmarks/results/jax_derivatives_cpu.json
```

Choose an available CPU with `--cpu`; the default is the first CPU in the
process affinity. Additional switches select workloads, backends, iteration
count, and sample count. PyTorch must be the first backend because it supplies
the numerical reference. Workers run sequentially to avoid competing for the
benchmark CPU. Timing thresholds are not enforced in CI; the numerical tests
run on Python 3.11 and 3.12. The recorded timing data uses Python 3.12.

The raw files include every sample, framework versions, numerical errors,
compile times, memory measurements, and reconstruction losses:

- {download}`Projection and reconstruction measurements <../benchmarks/results/jax_cpu.json>`
- {download}`JVP and HVP measurements <../benchmarks/results/jax_derivatives_cpu.json>`

JAX's [benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html)
explains synchronization and compilation accounting. Its
[custom derivative guide](https://docs.jax.dev/en/latest/301/custom-jvp-vjp.html)
describes automatic transposition of a linear JVP, and its
[checkpointing guide](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
explains the recomputation/storage tradeoff used by the scan variants.
