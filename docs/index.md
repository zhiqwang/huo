# Huo Documentation

Huo provides fan-beam CT forward projection and iterative image reconstruction,
with a Python / PyTorch implementation and a TypeScript / jax-js browser demo.

- [Tutorial](tutorial.md): forward projection, reconstruction, and the browser demo.
- [Geometry](geometry.md): scanner diagrams, coordinate conventions, sampling
  grids, and parameter reference.
- [Interactive Browser Demo](demo/index.md):
  watch forward projection and ART reconstruction run in your browser.
- [Residual Backprojection: Proofs and Limits](backward_propagation_proof.md):
  geometry, the connection to ART, convergence conditions, implementation findings,
  and executable Lean proofs.
- [Maintaining and Publishing Documentation](publishing.md): local previews,
  mathematical notation, proof checks, and GitHub Actions deployment.

Start with the tutorial to use the project, and use Geometry to check the
coordinate conventions and parameter meanings. For the mathematical justification
of `backprojection_angle`, the proof article states the assumptions and separates
continuous geometry from the sampled implementation.

Source code, installation instructions, and issue tracking are available in the
[GitHub repository](https://github.com/zhiqwang/huo).

```{toctree}
:hidden:
:maxdepth: 2

Tutorial <tutorial>
Geometry <geometry>
Interactive Browser Demo <demo/index>
Residual Backprojection: Proofs and Limits <backward_propagation_proof>
Maintaining and Publishing Documentation <publishing>
```
