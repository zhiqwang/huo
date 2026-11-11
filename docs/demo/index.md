# Interactive Browser Demo

Watch a phantom image become a sinogram, then reconstruct it by minimizing
projection loss with Adam. The computation runs in your browser using native
jax-js automatic differentiation and the same sampled model as Python.

## Run the demo

Select **Start** to project and reconstruct. **Stop** cancels the current solve;
**Start** begins a new solve from zero. Use **Frame delay** to adjust how long
each progress snapshot stays on screen.

```{include} ../../packages/demo/index.html
:start-after: <!-- demo-content:start -->
:end-before: <!-- demo-content:end -->
```

## Reading the results

- **Original phantom** is the input image, sampled on a 64 × 64 pixel grid.
- **Sinogram** shows 60 projection angles as rows and 80 detector positions
  as columns.
- **Adam reconstruction** starts from zero and fits the complete projection
  loss for 120 updates with learning rate 0.03 and nonnegative clipping.
- **Loss** is half mean squared residual, evaluated at the displayed image.

The same image and geometry are used for both stages. Gradients are evaluated
in chunks of eight angles, accumulated, and then used for one Adam update.
Progress snapshots appear every five updates. Repeated runs use the same
objective and starting point; floating-point results can vary across devices.

See the [jax-js guide](../jax_js.md) for the TypeScript API and ownership rules,
the [tutorial](../tutorial.md#browser-demo) for local setup, or [Geometry](../geometry.md) for scanner coordinates and
sampling conventions. The demo uses [jax-js](https://github.com/ekzhang/jax-js),
and Optax loaded from `esm.sh`. It uses WebGPU when available and WASM otherwise.
