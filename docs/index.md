# Huo Documentation

Huo provides differentiable fan-beam CT projection and loss-based image
reconstruction with PyTorch, Python JAX and browser jax-js. Optimize an image or a neural network
using a forward model, scalar loss, automatic gradients, and an optimizer.

- [Tutorial](tutorial.md): image optimization, batches, neural networks, and CLI.
- [JAX guide](jax.md): the optional JAX backend, PyTree state, JIT and Optax.
- [jax-js guide](jax_js.md): native browser gradients, ownership and chunked Adam.
- [Python API](api.md): geometry, the projector module, and the reconstruction helper.
- [Autograd Reconstruction](autograd.md): sampled forward models, gradients,
  higher derivatives, and memory.
- [JAX Comparison](jax_benchmark.md): numerical reproduction, CPU timings,
  compilation costs, and memory measurements.
- [Geometry](geometry.md): scanner diagrams, coordinate conventions, sampling
  grids, and parameter reference.
- [Interactive Browser Demo](demo/index.md):
  TypeScript / jax-js loss optimization with progress and cancellation.
- [Residual Backprojection: Proofs and Limits](backward_propagation_proof.md):
  historical residual-backfill analysis and its executable Lean proofs.
- [Maintaining and Publishing Documentation](publishing.md): local previews,
  mathematical notation, proof checks, and GitHub Actions deployment.

Start with the tutorial to use the project, and use Geometry to check the
coordinate conventions and parameter meanings. The autograd article derives
the shared data-loss gradient. The residual-backprojection article separately
analyzes the former browser ART algorithm.

Source code, installation instructions, and issue tracking are available in the
[GitHub repository](https://github.com/zhiqwang/huo).

```{toctree}
:hidden:
:maxdepth: 2

Tutorial <tutorial>
JAX guide <jax>
jax-js guide <jax_js>
Python API <api>
Autograd Reconstruction <autograd>
JAX Comparison <jax_benchmark>
Geometry <geometry>
Interactive Browser Demo <demo/index>
Historical ART: Proofs and Limits <backward_propagation_proof>
Maintaining and Publishing Documentation <publishing>
```
