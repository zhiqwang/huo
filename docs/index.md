# Huo Documentation

Huo provides differentiable fan-beam CT projection and loss-based image
reconstruction with PyTorch. Optimize an image or a neural network using a
standard forward, loss, backward, and optimizer loop.

- [Tutorial](tutorial.md): image optimization, batches, neural networks, and CLI.
- [Python API](api.md): geometry, the projector module, and the reconstruction helper.
- [Autograd Reconstruction](autograd.md): sampled forward models, gradients,
  higher derivatives, and memory.
- [Geometry](geometry.md): scanner diagrams, coordinate conventions, sampling
  grids, and parameter reference.
- [Interactive Browser Demo](demo/index.md):
  a separate TypeScript / jax-js example of ART-style residual backfill.
- [Residual Backprojection: Proofs and Limits](backward_propagation_proof.md):
  mathematical assumptions, limits, and executable Lean proofs for the browser
  algorithm.
- [Maintaining and Publishing Documentation](publishing.md): local previews,
  mathematical notation, proof checks, and GitHub Actions deployment.

Start with the tutorial to use the project, and use Geometry to check the
coordinate conventions and parameter meanings. The autograd article derives
the Python data-loss gradient. The residual-backprojection article separately
analyzes the browser algorithm.

Source code, installation instructions, and issue tracking are available in the
[GitHub repository](https://github.com/zhiqwang/huo).

```{toctree}
:hidden:
:maxdepth: 2

Tutorial <tutorial>
Python API <api>
Autograd Reconstruction <autograd>
Geometry <geometry>
Interactive Browser Demo <demo/index>
Browser ART: Proofs and Limits <backward_propagation_proof>
Maintaining and Publishing Documentation <publishing>
```
