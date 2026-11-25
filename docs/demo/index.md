# Interactive Browser Demo

Watch a phantom image become a sinogram, then reconstruct it by minimizing
projection loss with Adam. The computation runs in your browser using native
jax-js automatic differentiation and the same sampled model as Python.

## Run the demo

Choose **Adam updates** to run 120, 300, or 600 updates. The default is 300.
Select **Start** to project and reconstruct from zero. **Stop** cancels the
current solve. Use **Frame delay** to adjust how long each progress snapshot
stays on screen.

```{include} ../../packages/demo/index.html
:start-after: <!-- demo-content:start -->
:end-before: <!-- demo-content:end -->
```

## Reading the results

- **Original phantom** is the input image, sampled on a 64 × 64 pixel grid.
- **Sinogram** shows 60 projection angles as rows and 80 detector positions
  as columns.
- **Adam reconstruction** starts from zero and fits the complete projection
  loss with learning rate 0.03 and nonnegative clipping.
- **Projection loss** is half the mean squared difference between the measured
  sinogram and the projection of the displayed reconstruction.
- **Image error** compares the displayed reconstruction with the original
  phantom. It is the L2 norm of the pixel differences divided by the L2 norm
  of the phantom, expressed as a percentage. The zero image starts at 100%,
  and an exact match has 0% error.

The same image and geometry are used for both stages. Gradients are evaluated
in chunks of eight angles, accumulated, and then used for one Adam update.
Progress snapshots appear every five updates. The error curve records each
snapshot. Individual image errors can fluctuate because Adam minimizes the
projection loss. On WASM with the demo geometry, image error reached 21.7%
at 120 updates, 11.0% at 300, and 5.5% at 600. Floating-point results can vary
across devices.

See the [jax-js guide](../jax_js.md) for the TypeScript API and ownership rules,
the [tutorial](../tutorial.md#browser-demo) for local setup, or [Geometry](../geometry.md) for scanner coordinates and
sampling conventions. The site serves [jax-js](https://github.com/ekzhang/jax-js)
and Optax with the demo. It uses WebGPU when available and WASM otherwise.
