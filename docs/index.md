# Huo Documentation

Huo provides fan-beam CT forward projection and iterative image reconstruction,
with a Python / PyTorch implementation and a TypeScript / jax-js browser demo.

- [Tutorial](tutorial.md): geometry, forward projection, reconstruction, and the browser demo.
- [Residual Backprojection: Proofs and Limits](backward_propagation_proof.md):
  geometry, the connection to ART, convergence conditions, implementation findings,
  and executable Lean proofs.
- [Maintaining and Publishing Documentation](publishing.md): local previews,
  mathematical notation, proof checks, and GitHub Actions deployment.

Start with the tutorial to use the project. For the mathematical justification
of `backward_propagation`, the proof article states the assumptions and separates
continuous geometry from the sampled implementation.

Source code, installation instructions, and issue tracking are available in the
[GitHub repository](https://github.com/zhiqwang/huo).

```{toctree}
:hidden:
:maxdepth: 2

Tutorial <tutorial>
Residual Backprojection: Proofs and Limits <backward_propagation_proof>
Maintaining and Publishing Documentation <publishing>
```
