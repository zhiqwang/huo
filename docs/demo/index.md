# Interactive Browser Demo

Watch a phantom image become a sinogram, then reconstruct it with Adam and
total variation (TV). The computation runs in your browser using native
jax-js automatic differentiation and the same sampled model as Python.

## Run the demo

Choose **Adam updates** to run 120, 300, or 600 updates. The default is 300.
**TV weight** defaults to 0.5 for this phantom. Select **Off** to compare with
projection-only Adam on the same data.
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
  objective with learning rate 0.03 and nonnegative clipping.
- **Projection loss** is half the mean squared difference between the measured
  sinogram and the projection of the displayed reconstruction.
- **Objective** adds the weighted TV penalty to projection loss. TV is the sum
  of mean absolute differences along the two image axes.
- **Relative L2** is the L2 norm of the pixel differences divided by the L2 norm
  of the phantom, expressed as a percentage. A zero image starts at 100%.
- **SSIM** compares local brightness, contrast and structure using an 11 × 11
  Gaussian window with standard deviation 1.5 pixels, averaged over windows
  contained in the image. **PSNR** expresses the squared image error on a
  decibel scale. Both use the phantom's intensity range.
- **Flat-region variation** is the RMS difference between neighbouring pixels
  in positive flat parts of the phantom, excluding its edges and their
  one-pixel neighbourhood. It is expressed as a percentage of the phantom's peak.
  A constant image has zero variation, so read this alongside L2 and SSIM.

The data gradient is accumulated in chunks of eight angles. The TV gradient is
added once before each Adam update. Every five updates, the demo evaluates the
losses and image metrics at the displayed snapshot. The chart tracks relative L2.

Adding TV to the data term follows the open-source TV-Adam examples. The demo
uses the known phantom reference to compare image quality.
[DIVαℓ TV-Adam](https://github.com/jleuschn/dival/blob/master/dival/reconstructors/tvadam_ct_reconstructor.py)
· [DiffVox](https://arxiv.org/html/2411.19224v2#S2)
· [SSIM implementation settings](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity)

See the [jax-js guide](../jax_js.md) for the TypeScript API and ownership rules,
the [tutorial](../tutorial.md#browser-demo) for local setup, or [Geometry](../geometry.md) for scanner coordinates and
sampling conventions. The site serves [jax-js](https://github.com/ekzhang/jax-js)
and Optax with the demo. It uses WebGPU when available and WASM otherwise.
