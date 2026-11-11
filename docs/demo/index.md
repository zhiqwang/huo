# Interactive Browser Demo

Watch a phantom image become a sinogram, then reconstruct it with the
Algebraic Reconstruction Technique (ART). The computation runs in your browser.

This TypeScript example uses uniform residual backfill. The Python package
uses [loss-based optimization](../autograd.md); the two implementations share
forward geometry and use different reconstruction updates.

## Run the demo

Select **Start** to run forward projection and ART. Use **Frame delay** to
adjust the animation speed.

```{include} ../../packages/demo/index.html
:start-after: <!-- demo-content:start -->
:end-before: <!-- demo-content:end -->
```

## Reading the results

- **Original phantom** is the input image, sampled on a 128 × 128 pixel grid.
- **Sinogram** shows 180 projection angles as rows and 200 detector positions
  as columns. Each row appears as its forward projection completes.
- **ART reconstruction** starts from zero and updates the image by
  backprojecting the residual for each angle in random order.

The same image and geometry are used for both stages. Select **Start** again
to repeat the reconstruction with a new angle order.

See the [tutorial](../tutorial.md#browser-demo) for local setup and
the TypeScript API, or [Geometry](../geometry.md) for scanner coordinates and
sampling conventions. The demo uses [jax-js](https://github.com/ekzhang/jax-js),
loaded from `esm.sh`.
