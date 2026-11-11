# Autograd Reconstruction

Huo's Python reconstruction minimizes a scalar loss using PyTorch. The image
or neural-network weights are optimization variables; acquisition geometry is
fixed. The [tutorial](tutorial.md) contains runnable loops.

## The sampled forward model

Let $x\in\mathbb R^{N^2}$ denote one flattened image. Fixed ray positions,
bilinear interpolation with zero padding, and ray-step summation define a
linear map

$$
A:\mathbb R^{N^2}\to\mathbb R^{VM}.
$$

`FanbeamProjector(image)` evaluates $Ax$ without assembling a matrix. Each
row of the resulting `[V, M]` sinogram corresponds to an angle in the supplied
order. Batch dimensions apply the same map independently to each image.

Given fixed measured line integrals $y$, the helper minimizes

$$
L(x)=\frac{1}{2VM}\lVert Ax-y\rVert_2^2.
$$

For a batch of $Q$ images, the loss averages over $QVM$ measurements. The
gradient for each image therefore includes an additional factor $1/Q$.

## What backward computes

The chain rule gives

$$
\nabla_x L=\frac{1}{VM}A^T(Ax-y).
$$

The autograd backward applies the transpose of the *actual sampled weights*,
including ray-step scale, interpolation weights, and partial zero-padding
weights at image boundaries. It receives the loss gradient at the sinogram;
it does not reconstruct an image by simply backfilling measured intensities.

```python
predicted = projector(image)
loss = 0.5 * (predicted - measured).square().mean()
loss.backward()       # writes the gradient into image.grad
optimizer.step()     # updates image using that gradient
```

The public interface only needs forward projection. `_ProjectTranspose` is a
private implementation of $A^T$ used during backward. It calls the native
bilinear sampling VJP from the pinned PyTorch version directly, avoiding a
forward sampling pass over a synthetic zero image.

## Higher derivatives and memory

`_Project` and `_ProjectTranspose` form a pair: the backward of $A$ applies
$A^T$, and the backward of $A^T$ applies $A$. Their JVP rules apply the same
linear map to input tangents. A quadratic data loss has

$$
\nabla_x^2 L=\frac{1}{VM}A^T A.
$$

This allows Hessian-vector products without assembling the Hessian or saving
every view's interpolation indices. All views share one autograd node and the
zero-angle ray coordinates. Rotated sampling grids are recomputed as needed.
The retained sampling constants scale with the ray count and number of angles,
not with a separate image sampling grid for every angle and batch element.

The regression suite compares gradients with matrices constructed from basis
images, checks finite differences, exercises `torch.func` compositions, and
bounds retained sampling storage. `benchmarks/autograd.py` reports timings and
unique live saved-tensor storage for ordinary and higher-order gradient graphs.

## Constraints and optimizers

With plain SGD and step size $\eta$, a nonnegative projected update is

$$
x^{k+1}=P_+\left(x^k-\eta\nabla_x L(x^k)\right).
$$

The convenience solver uses Adam, whose moment estimates and adaptive scaling
determine the update before the nonnegative clamp. `lr` and `steps` specify
optimizer settings; they do not certify convergence or image uniqueness.
The helper returns a detached result. The projector remains differentiable
when included in a custom optimization loop or neural network.

Regularizers can be added to the loss using ordinary PyTorch operations. For
example, a finite-difference smoothness term contributes its own gradient in
addition to the projection-data gradient. With incomplete or noisy data, the
chosen penalty and constraints influence which image is recovered.

## Relationship to the browser ART example

The TypeScript browser example implements a separate uniform residual-backfill
map $B_\theta$ and updates one angle at a time. Its $B_\theta$ is generally not
$A_\theta^T$. Python loss optimization uses the transpose of its sampled forward
map and does not reproduce that browser update sequence.

The [residual-backprojection article](backward_propagation_proof.md) and Lean
proofs analyze that separate algorithm and its mathematical assumptions. A
successful gradient check establishes differentiation of the sampled model;
it does not establish that a measurement set uniquely determines an image.
