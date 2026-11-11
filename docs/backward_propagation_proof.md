---
# Keep the narrow page sidebar limited to main sections.
tocdepth: 2
---

# Historical ART: Residual Backprojection and Its Limits

Residual backprojection can correct projection errors, but it is not an inverse
of forward projection. This historical article reviews the former browser operators in
[the archived art.ts](https://github.com/zhiqwang/huo/blob/37d1e524df2d3c82fcbe6942b466903eb16405ff/packages/src/art.ts), derives
conditions under which their *idealized iterations* converge, and identifies
what those conditions do and do not establish about that implementation.

The main distinctions are between a continuous ray model and sampled matrices,
between one ray and a whole angle, and between a small projection residual and
recovery of a particular image. That implementation performed one
randomly ordered sweep from zero. None of the convergence theorems below
certifies exact reconstruction by that single sweep.

The current Python and [jax-js](jax_js.md) packages use a forward-only
`FanbeamProjector` and a loss/optimizer loop. Their backward applies the sampled
transpose automatically. The former ART/backfill methods have been removed. See
[Autograd Reconstruction](autograd.md) for that objective and gradient; the
iteration analyzed here is the historical residual-backfill algorithm, not the
current browser Adam solver. The Lean statements remain mathematical results
about their stated models.

All mathematical statements use real arithmetic. Discretization error,
measurement noise, and floating-point error are separate issues. Selected
algebraic and convergence results have executable Lean proofs; see
[Lean proofs](#lean-proofs) for the exact coverage.

(the-update-implemented-by-the-code)=
(update-rule)=
## Iterative update

Use the notation in [Geometry](geometry.md): $N\geq2$ image pixels per
axis, $M\geq2$ detector elements, $V$ angles in radians, and pixel-length
units. Let $n=N^2$ and $L=N$. Assume positive finite scanner distances and
detector spacing, a positive integer sampling multiplier, and a positive
perspective denominator at every backprojected pixel. The mathematical
statements below concern geometries for which these operations are defined.

Flatten the image into $x\in\mathbb R^n$ and define the fixed linear maps

$$
A_\theta:\mathbb R^n\to\mathbb R^M,
\qquad B_\theta:\mathbb R^M\to\mathbb R^n.
$$

They represent forward projection and `backprojection_angle`, respectively.
Interpolation at fixed coordinates and finite summation are linear in their
input values. Thus matrices exist even though the code does not store them.
This is a real-arithmetic model of the operations, not exact linearity of
floating-point evaluations.

| Symbol | Meaning | Code |
| --- | --- | --- |
| $x$ | Current image | `image` |
| $b_\theta$ | Measured projection at one angle | `sinogram[j, :]` |
| $A_\theta x$ | Predicted projection | `forward_angle(...)` |
| $r_\theta=b_\theta-A_\theta x$ | Residual before normalization | `sinogram[j, :] - predicted` |
| $B_\theta r_\theta$ | Backprojected residual | `backprojection_angle(...)` |
| $L=N$ | Ray-window length used for normalization | `geometry.resolution` |
| $P_+z$ | Componentwise $\max(z_j,0)$ | `Math.max(0, value)` |

Each angle update is therefore

$$
\boxed{x^{k+1}=P_+\!\left[x^k+\alpha B_{\theta_k}
(b_{\theta_k}-A_{\theta_k}x^k)\right],\qquad \alpha=1/L.}
\tag{1}
$$

Later results allow a general positive $\alpha$. In (1), $B_\theta$ acts on a
*residual*. Its name does not imply $B_\theta=A_\theta^{-1}$ or
$B_\theta=A_\theta^T$.

(what-the-continuous-geometry-proves)=
## Ray geometry

The coordinate system and symbols are shared with [Geometry](geometry.md).
Columns follow x and rows follow y. The source is $S=(0,d_s)$, the detector
plane is $y=-d_d$, and $D=d_s+d_d$. All distances are in pixels. A public
angle $\theta$ is in radians: forward projection rotates scanner rays by
$R_{-\theta}$, and backprojection uses $R_\theta$ to return image pixels to
scanner coordinates.

### The detector coordinate is constant along a ray

For an image point $p=(x,y)$, set $(x',y')=R_\theta p$ and assume $d_s-y'>0$.
Intersecting its ray with the detector plane gives

$$
u_\theta(p)=\frac{D x'}{d_s-y'}.
\tag{2}
$$

**Proposition 1.** Every point on the forward ray toward detector $u$ maps to $u$.

**Proof.** Parameterize the ray by distance $t>0$ from the source:

$$
\begin{aligned}
x'(t)&=\frac{tu}{\sqrt{u^2+D^2}},\\
y'(t)&=d_s-\frac{tD}{\sqrt{u^2+D^2}}.
\end{aligned}
$$

Substituting into (2) cancels the nonzero common factors. The Lean theorem
`ray_coordinate` checks this identity with `source_distance` $=d_s$,
`det_distance` $=d_d$, and `scale` $=t/\sqrt{u^2+D^2}$. $\square$

The pixel-centre extent is $a_x=(N-1)/2$, named `pixel_extent` in both
languages. For normalized scanner coordinates $\xi=x'/a_x$, $\eta=y'/a_x$,

$$
\frac{D\xi}{d_s/a_x-\eta}=\frac{D x'}{d_s-y'}.
$$

Dividing the detector coordinate by $a_u=(M-1)\Delta u/2$, named
`det_extent`, gives its normalized interpolation coordinate. Equivalently,
the fractional detector index is

$$
\frac{M-1}{2}\left(\frac{u_\theta(p)}{a_u}+1\right)
=\frac{u_\theta(p)}{\Delta u}+\frac{M-1}{2}.
$$

The historical browser implementation used the right-hand expression to interpolate the 1-D
detector row directly. This verifies the perspective formula on the shared
grid, not an adjoint identity.

### Backfill followed by integration reproduces a scaled projection

Fix a bounded detector interval $U$. Let $\Omega$ be a bounded measurable
region in front of the source, restricted to points with $u_\theta(p)\in U$.
For each $u$, let $\Gamma_{\theta,u}$ be the portion of its ray actually
integrated, including any integration-window restriction. Assume these
segments form a measurable integration domain. Work with bounded Borel
functions for which the following integrals exist, rather than arbitrary
representatives of area-almost-everywhere equivalence classes.

Define

$$
(\mathcal A_\theta f)(u)=\int_{\Gamma_{\theta,u}} f(p)\,dt,
\qquad
(\mathcal B_\theta g)(p)=g(u_\theta(p)),
\qquad
\ell_\theta(u)=\int_{\Gamma_{\theta,u}}1\,dt.
\tag{3}
$$

**Proposition 2.** On these rays,

$$
\boxed{(\mathcal A_\theta\mathcal B_\theta g)(u)
=\ell_\theta(u)g(u).}
\tag{4}
$$

**Proof.** Proposition 1 makes $g(u_\theta(p))=g(u)$ constant on the
integration path. Pulling that constant out of the integral gives (4).
$\square$

Let $\mathcal D_\theta$ multiply by $\ell_\theta(u)$. On rays with positive
length, provided division by length preserves the operator domain,

$$
\mathcal C_\theta=\mathcal B_\theta\mathcal D_\theta^{-1}
\quad\Longrightarrow\quad \mathcal A_\theta\mathcal C_\theta=I.
\tag{5}
$$

A uniform lower bound $\ell_\theta(u)\geq\ell_{\min}>0$ suffices for
bounded data. This is a **right inverse**: backfill a normalized projection,
then integrate, and recover that projection. It does not imply
$\mathcal C_\theta\mathcal A_\theta=I$ on images. Whenever a ray has a
nontrivial interval of support, a function can vary along it while having
zero integral. Such changes are invisible to that angle.

### The residual contracts under a length condition

**Proposition 3.** Without clipping, let $r=b_\theta-\mathcal A_\theta f$
and $f^+=f+\mathcal B_\theta r/L$. Then

$$
\boxed{r^+(u)=\left(1-\frac{\ell_\theta(u)}{L}\right)r(u).}
\tag{6}
$$

**Proof.** Linearity and (4) give
$r^+=r-\mathcal D_\theta r/L$. $\square$

A ray with $\ell_\theta(u)=L$ has its residual eliminated in one update.
For $0<\ell_\theta(u)/L<2$, its residual magnitude strictly decreases
unless already zero. For a uniform $L^2(U,du)$ contraction, assume

$$
0<\ell_{\min}\leq\ell_\theta(u)\leq\ell_{\max}<2L
\quad\text{almost everywhere},\qquad r\in L^2(U,du).
$$

Then

$$
\|r^+\|_{L^2}\leq q\|r\|_{L^2},\qquad
q=\max\!\left\{\left|1-\frac{\ell_{\min}}L\right|,
\left|1-\frac{\ell_{\max}}L\right|\right\}<1.
\tag{7}
$$

Indeed, the absolute value of an affine function attains its maximum on an
interval at an endpoint; square (6), integrate, and take square roots.
Pointwise strict decrease alone does not supply a uniform $q<1$ when lengths
approach zero. A zero-length ray cannot have a nonzero measurement corrected
by an image update.

Using $\mathcal C_\theta r$ instead eliminates all residuals of this angle
in one step, but can change projections at other angles. This is not a
multi-angle convergence theorem. Lean verifies the scalar algebra and strict
length condition in `length_residual` and `length_contraction`; it does not
formalize the ray integral or this $L^2$ argument.

### Uniform fan-beam backfill is not the usual continuous adjoint

This distinction exists even before discretization. For the parameterization
$p(t,u)$ above, the area Jacobian is

$$
J(t,u)=\left|\det\frac{\partial p}{\partial(t,u)}\right|
=\frac{t\,D}{u^2+D^2}>0,
\qquad dp=J(t,u)\,dt\,du.
\tag{8}
$$

For example, assume a compact integration domain with $t$ bounded away from
zero, so the following change of variables is valid and all weights are
bounded. With image measure $dp$ and detector measure $du$,

$$
\begin{aligned}
\langle\mathcal A_\theta f,g\rangle_{L^2(du)}
&=\int_U\int_{\Gamma_{\theta,u}} f(p(t,u))g(u)\,dt\,du\\
&=\int_{\Omega_{\rm int}}f(p)\,
\frac{g(u_\theta(p))}{J(t(p),u_\theta(p))}\,dp.
\end{aligned}
$$

Here $\Omega_{\rm int}$ is the region covered by the selected integration
segments. Thus the adjoint on this region is $\mathcal B_\theta g/J$, with
zero contribution outside it if the image space is larger. Uniform backfill
omits this weight. Changing the image inner product can change the adjoint,
but the chosen measure must be stated; this Jacobian also depends on the
angle when written in world coordinates. A right inverse, an adjoint, and
an image inverse are distinct objects.

(the-precise-connection-to-art)=
## ART connection

### A single nonzero row

Let $a_i$ be a nonzero row of a discrete forward matrix and
$r_i=b_i-a_ix$. The minimum-change correction solves

$$
\min_\delta\tfrac12\|\delta\|_2^2
\quad\text{subject to}\quad a_i\delta=r_i.
$$

Stationarity of the Lagrangian gives $\delta=\nu a_i^T$; feasibility gives
$\nu=r_i/\|a_i\|_2^2$. Strict convexity establishes the unique minimizer:

$$
\boxed{\delta=\frac{r_i}{\|a_i\|_2^2}a_i^T.}
\tag{9}
$$

This is the orthogonal projection onto $a_ix=b_i$. See
[Kak and Slaney, Chapter 7, Sections 7.1–7.2](https://engineering.purdue.edu/~malcolm/pct/CTI_Ch07.pdf)
for the classical ART/Kaczmarz construction.

If the ray visits $N_i>0$ pixels, each with weight $h>0$, its row norm is
$N_i h^2$ and its total length is $L_i=N_i h$. Within its support, (9) becomes

$$
\delta_j=\frac{r_i h}{N_i h^2}=\frac{r_i}{L_i}.
\tag{10}
$$

Thus uniform residual backfill is exactly ART **in this equal-weight model**.
Replacing $L_i$ by a common $L$ gives relaxed ART with $\lambda_i=L_i/L$.
For any consistent solution $x_*$ and $e=x-x_*$,

$$
\boxed{\|e^+\|_2^2=\|e\|_2^2-
\lambda_i(2-\lambda_i)\frac{|a_i e|^2}{\|a_i\|_2^2}.}
\tag{11}
$$

**Proof.** Since $r_i=-a_i e$, expand the squared norm of
$e-\lambda_i(a_i e)a_i^T/\|a_i\|_2^2$ and collect terms. $\square$

For $0<\lambda_i<2$, distance to every consistent solution decreases when
this row has nonzero residual. Lean proves (10) and (11) as
`equal_weight_art` and `art_energy`, including the nonincrease consequence
`art_energy_nonincreasing`.

### A whole angle is generally not one Kaczmarz step

The implementation updates all detector residuals at once. If the rows for
one angle have disjoint pixel supports, their corrections do not interfere,
and simultaneous and sequential updates agree. Interpolation generally
creates overlapping supports, invalidating this equivalence.

Even when $B_\theta=A_\theta^T$, simultaneous updating is a block gradient
step, not usually the orthogonal projection onto the block constraints.
For a full-row-rank block, that projection instead uses

$$
x^+=x+A_\theta^T(A_\theta A_\theta^T)^{-1}(b_\theta-A_\theta x).
\tag{12}
$$

For a rank-deficient consistent block, use the corresponding pseudoinverse.
A scalar division by $L$ does not generally implement this matrix inverse.

(the-actual-sampled-operators)=
## Discrete model

Let $K=Nq_{\rm ray}$, where $q_{\rm ray}$ is `ray_sampling`, so $\Delta t=L/K=1/q_{\rm ray}$. The code samples
$K+1$ points, including both endpoints of the distance interval
$[d_s-L/2,d_s+L/2]$. If $z_{\theta,i,k}$ is the pixel-coordinate
position of forward sample $k$ and $\varphi_j$ its bilinear weight for image
pixel $j$, then

$$
(A_\theta)_{ij}=\Delta t\sum_{k=0}^{K}\varphi_j(z_{\theta,i,k}).
\tag{13}
$$

The weights include the code's zero-padding convention. If $p_j$
is the shared image-pixel location and $\psi_i$ is the one-dimensional
detector interpolation weight, then

$$
(B_\theta)_{ji}=\psi_i(u_\theta(p_j)).
\tag{14}
$$

Both formulas follow by expanding the interpolated values as finite weighted
sums. They are different weight constructions. Neither transpose matching nor
the row norm normalization in (9) follows from their geometry.

### Image length is not an exact discrete normalization

The forward rule gives full weight to both endpoints. A constant value of
one at **every sampled point** would yield
$(K+1)\Delta t=L+\Delta t$, not $L$. Actual samples may taper or vanish
because of zero padding, and the distance window may omit part of the
image support. In particular, a square image can have chords longer than
$L$; a circular support mask was not enforced by the historical browser operators.
Thus neither a full chord length nor $L$ automatically equals the actual
row sum

$$
\ell^{\rm disc}_{\theta,i}=(A_\theta\mathbf1)_i
=\Delta t\sum_{k,j}\varphi_j(z_{\theta,i,k}).
\tag{15}
$$

Even normalizing by (15) would not prove that $A_\theta B_\theta$ is diagonal.
Image and detector interpolation mix neighboring rays. Samples just outside
the pixel-center range can still receive partial weight from an edge pixel;
zero padding does not mean that every such sample vanishes immediately.
The interpolation and padding conventions are documented in
[PyTorch's `grid_sample` reference](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.grid_sample.html).

### Exact discrete residual and error identities

For an unclipped update, linearity gives

$$
\boxed{r_\theta^+=(I-\alpha A_\theta B_\theta)r_\theta.}
\tag{16}
$$

If $b_\theta=A_\theta x_*$, image error instead evolves as

$$
\boxed{e^+=(I-\alpha B_\theta A_\theta)e=:T_\theta e.}
\tag{17}
$$

Lean checks these for arbitrary real linear maps in `residual_identity` and
`error_identity`. The multiplication order matters: (16) acts on detector
space, and (17) acts on image space. A zero residual leaves nullspace image
errors undetected. The continuous diagonal identity (4) cannot be substituted
for the sampled matrix product in (16).

(convergence-with-fixed-linear-operators)=
## Convergence

### Recovering a specified solution from every initial image

**Theorem 1.** Fix $A\in\mathbb R^{m\times n}$,
$B\in\mathbb R^{n\times m}$, $\alpha>0$, and consistent data $b=Ax_*$.
For the unclipped iteration

$$
x^{k+1}=x^k+\alpha B(b-Ax^k),
\tag{18}
$$

convergence to this same $x_*$ from **every** $x^0$ holds if and only if

$$
\boxed{\rho(I-\alpha BA)<1.}
\tag{19}
$$

**Proof.** Set $T=I-\alpha BA$. Equation (17) yields
$e^k=T^k e^0$. If $\rho(T)<1$, put $T$ in complex Jordan form. Powers of
a block with nonzero eigenvalue $\lambda$ are sums of terms
$\binom{k}{j}\lambda^{k-j}N^j$ for finitely many fixed $j$.
For $|\lambda|<1$ each tends to zero; blocks with $\lambda=0$ are nilpotent.
Hence $T^k\to0$. Conversely, convergence for every real initial error implies
convergence on every standard basis vector, and therefore $T^k\to0$ entrywise.
A complex eigenvector with $|\lambda|\geq1$ would give
$T^kv=\lambda^kv\not\to0$, a contradiction. $\square$

For each complex eigenvalue $\lambda$ of $BA$,

$$
|1-\alpha\lambda|^2
=1-2\alpha\operatorname{Re}\lambda+\alpha^2|\lambda|^2.
$$

Thus (19) is equivalent to requiring, for **every** eigenvalue,

$$
\operatorname{Re}\lambda>0,\qquad
0<\alpha<\frac{2\operatorname{Re}\lambda}{|\lambda|^2}.
\tag{20}
$$

Zero eigenvalues cannot be discarded for this all-initial-images statement.
Also, $\rho(T)<1$ need not imply $\|T\|_2<1$: nonnormal matrices can amplify
errors transiently before eventual decay.

No step of this proof assumes $B=A^T$. For example, $A=I$, $B=2I$,
$\alpha=1/4$ gives $T=I/2$ and halves every error. For related theory of
unmatched projectors and restricted solution spaces, see
[Dong et al., Section 2, “The BA Iteration”](https://arxiv.org/html/1902.04282#S2).
The full-space hypothesis here is deliberately stronger than convergence in
a selected subspace.

### Rank deficiency and the solution actually selected

If $Az=0$, then $Tz=z$. A nonzero nullspace therefore rules out (19), but
does **not** rule out useful convergence to *some* consistent image.

For example, take $B=A^T$, $A\ne0$, consistent data, and
$0<\alpha<2/\|A\|_2^2$. Decompose the image space orthogonally as
$\operatorname{range}(A^T)\oplus\ker A$. On the first subspace, the
nonzero singular values $\sigma_j$ give factors
$1-\alpha\sigma_j^2$ of magnitude less than one; on the nullspace, $T$
is the identity. Therefore

$$
x^k\longrightarrow A^\dagger b+P_{\ker A}x^0.
\tag{21}
$$

Here $A^\dagger b$ is the minimum Euclidean norm solution and $P_{\ker A}$
is orthogonal projection. Starting at zero selects that minimum-norm solution,
which need not be the original image. Equation (21) concerns the fixed,
matched, unclipped iteration, not the historical browser's angle updates. For general
$B$, the updates remain in the affine space $x^0+\operatorname{range}(B)$;
restricted stability and solution selection require their own hypotheses.

Likewise, for consistent data and a full-column-rank stacked matrix,

$$
\|x-x_*\|_2\leq\frac{\|Ax-b\|_2}{\sigma_{\min}(A)}.
\tag{22}
$$

This follows from the smallest singular value bound. Without injectivity,
no such bound can control the whole image error: adding a nullspace vector
changes the image without changing its residual. Even with injectivity, a
small $\sigma_{\min}$ makes the inference poorly conditioned.

### Noise and bounded perturbations

Under (19), $BA$ is invertible. For arbitrary, possibly inconsistent $b$,
the fixed iteration converges to

$$
x_\infty=(BA)^{-1}Bb,\qquad B(Ax_\infty-b)=0.
\tag{23}
$$

**Proof.** This vector is a fixed point; subtract it from (18) and apply
$T^k\to0$. If $BA$ had a zero eigenvalue, $T$ would have eigenvalue one,
so invertibility follows from (19). $\square$

This is generally not the ordinary least-squares normal equation
$A^T(Ax-b)=0$. For $b=Ax_*+\varepsilon$,
$x_\infty-x_*=(BA)^{-1}B\varepsilon$, so stability does not eliminate
noise amplification.

A quantitative extension uses a norm contraction. If an update, or a whole
sweep, satisfies

$$
\|e^{k+1}\|\leq q\|e^k\|+\eta,\qquad 0\leq q<1,\quad\eta\geq0,
$$

then induction gives

$$
\boxed{\|e^k\|\leq q^k\|e^0\|+
\eta\frac{1-q^k}{1-q}.}
\tag{24}
$$

The limiting error bound is $\eta/(1-q)$. For a fixed linear update with
additive disturbance $w^k$, $\|T\|\leq q$ and $\|w^k\|\leq\eta$
imply the premise by the triangle inequality. A bound on real rounding or
modeling errors is an additional obligation, not an assumption verified by
the implementation. Lean checks (24) as `perturbed_error_bound`, and the
zero-disturbance bound and limit as `geometric_error_bound` and
`tendsto_of_error_contraction`.

The theorem `noisy_error_identity` verifies the additive term
$\alpha B\varepsilon$ for perturbed measurements. The theorem
`linear_iterates_converge` connects the exact recurrence (18) to convergence
under the sufficient bound $\|Te\|\leq q\|e\|$ for every $e$.

(multiple-angles-and-their-order)=
## Angle ordering

### Repeating a fixed order

**Theorem 2.** Assume $V\geq1$ fixed angles and a common solution
$b_i=A_i x_*$. Remove clipping and repeat the order $1,\ldots,V$.
Set

$$
T_i=I-\alpha B_iA_i,\qquad Q=T_V\cdots T_1.
\tag{25}
$$

Convergence to the same $x_*$ from every initial image holds if and only if
$\rho(Q)<1$.

**Proof.** Errors at sweep boundaries satisfy
$e^{[t+1]}=Qe^{[t]}$, so Theorem 1's matrix-power argument applies to $Q$.
Within a sweep, errors are fixed prefix products
$T_j\cdots T_1e^{[t]}$. Their norms have a finite common bound, so boundary
convergence implies convergence of all intermediate steps. The converse
follows by taking the boundary subsequence. $\square$

Products are ordered: the first update acts on the right. Replacing $Q$ by
$I-\alpha\sum_i B_iA_i$ describes a different algorithm. If a nonzero
$z$ belongs to every $\ker A_i$, then each $T_i z=z$ and $Qz=z$, so
full-space contraction is impossible. Lean checks the nullspace facts and
ordered sweep composition in `nullspace_error_fixed`, `sweep_common_fixed`,
and `sweep_append`.

With inconsistent data, an angle step is affine, $x^+=T_i x+c_i$, where
$c_i=\alpha B_i b_i$. A sweep has the form

$$
x^{[t+1]}=Qx^{[t]}+c,\qquad
c=\sum_{i=1}^V T_V\cdots T_{i+1}c_i,
\tag{26}
$$

with an empty product equal to $I$. If $\rho(Q)<1$, boundaries converge to
$(I-Q)^{-1}c$, but intermediate angle states can approach a periodic cycle.
There need not be a common fixed image or an ordinary least-squares solution.
The fixed-operator formula (23) cannot simply be reused for this schedule.

### Changing the order between sweeps

If sweep $t$ uses permutation $\pi_t$, then, for consistent unclipped updates,

$$
e^{[t]}=Q_{\pi_{t-1}}\cdots Q_{\pi_0}e^{[0]}.
\tag{27}
$$

A sufficient condition is a **common induced norm** and $0\leq q<1$ with
$\|Q_\pi\|\leq q$ for every allowed permutation. Submultiplicativity gives
$\|e^{[t]}\|\leq q^t\|e^{[0]}\|$.

Individual spectral radii alone do not suffice for arbitrary switching.
For example,

$$
W_1=\begin{pmatrix}1/2&2\\0&1/2\end{pmatrix},\quad
W_2=\begin{pmatrix}1/2&0\\2&1/2\end{pmatrix},\quad
W_2W_1=\begin{pmatrix}1/4&1\\1&17/4\end{pmatrix}.
$$

Both $W_1$ and $W_2$ have spectral radius $1/2$, but
$\rho(W_2W_1)=9/4+\sqrt5>1$. This demonstrates the logical gap in a switching
argument; it does not assert that these matrices arise from Huo's angles.
A random permutation of angle blocks is also not the same sampling scheme
as independently randomized, row-norm-weighted Kaczmarz updates, so that
method's convergence results cannot be imported without checking hypotheses.

(what-nonnegative-clipping-changes)=
## Clipping

### Distance to a nonnegative solution

**Lemma.** For $x_*\geq0$,

$$
\|P_+(z)-x_*\|_2\leq\|z-x_*\|_2.
\tag{28}
$$

**Proof.** If $z_j\geq0$, the coordinate is unchanged. Otherwise,
$|0-x_{*,j}|=x_{*,j}\leq x_{*,j}-z_j=|z_j-x_{*,j}|$.
Square and sum over the coordinates. $\square$

Lean proves the scalar and finite-sum statements in `clamp_sq_le` and
`clamp_sum_sq_le`. The latter explicitly uses a sum of squares: Lean's plain
function-space norm on finite vectors is a supremum norm, not automatically
the Euclidean norm.

Equation (28) compares the candidate *before and after clipping*. It does
not compare with the previous iterate and does not establish residual
monotonicity. It also does not let us use Theorem 2 unchanged: clipping is
inserted between the factors of its matrix product.

### A sufficient condition for clipped sweeps

**Theorem 3.** Suppose there is a common nonnegative solution $x_*$. Let

$$
\mathscr D=\{\operatorname{diag}(d_1,\ldots,d_n):0\leq d_j\leq1\}.
$$

If there exists $0\leq q<1$ such that every allowed permutation $\pi$ and
every choice $D_1,\ldots,D_V\in\mathscr D$ satisfy

$$
\boxed{\|D_VT_{\pi(V)}\cdots D_1T_{\pi(1)}\|_2\leq q,}
\tag{29}
$$

then the clipped iteration converges to $x_*$ from every initial image, with
$\|e^{[t]}\|_2\leq q^t\|e^{[0]}\|_2$ at sweep boundaries.

**Proof.** The scalar function $h(a)=\max(a,0)$ is monotone and
$1$-Lipschitz. Hence $h(a)-h(b)=d(a-b)$ for some $d\in[0,1]$:
use the difference quotient when $a\ne b$, and $d=0$ otherwise.
The unclipped candidate is $z=x_*+T_i(x-x_*)$. Apply this scalar identity
to each coordinate and use $P_+(x_*)=x_*$ to obtain

$$
e^{k+1}=D_kT_i e^k,\qquad D_k\in\mathscr D.
\tag{30}
$$

Multiply along a sweep and apply (29). Induction gives the geometric bound,
which tends to zero. The finitely many possible angle prefixes are bounded,
and $\|D_k\|_2\leq1$, so intermediate errors also tend to zero. $\square$

This is a conservative sufficient condition, not a necessary one. In
particular, it includes $D_i=I$ and is impossible on the full space if the
angles have a common nontrivial nullspace. The example acquisition dimensions
have that obstruction. Lean checks the scalar secant identity
`clamp_secant`, its coordinatewise extension `clipped_error_representation`,
and the generic norm-contraction bound and limit; the universal
matrix-product hypothesis (29) is **not** verified for Huo.

### Nonnegative weights can have a wrong nonnegative fixed point

Take

```{math}
:nowrap:

\begin{equation}
A=I_2,\qquad B=\begin{pmatrix}1&2\\2&1\end{pmatrix},\qquad
x_*=b=\begin{pmatrix}1\\2\end{pmatrix}.
\tag{31}
\end{equation}
```

The matrices have nonnegative entries and $Ax=b$ has a unique solution.
Nevertheless, $BA$ has eigenvalues $3$ and $-1$. The error direction
$(1,-1)^T$ is multiplied by $1+\alpha>1$ for every $\alpha>0$, so the
unclipped iteration cannot converge from every initial image.

Clipping does not repair the guarantee. At $\bar x=(5,0)^T$,

$$
b-A\bar x=\begin{pmatrix}-4\\2\end{pmatrix},\qquad
B(b-A\bar x)=\begin{pmatrix}0\\-6\end{pmatrix},
$$

and consequently

```{math}
:nowrap:

\begin{equation}
P_+\!\left[\bar x+\alpha B(b-A\bar x)\right]
=P_+\begin{pmatrix}5\\-6\alpha\end{pmatrix}
=\bar x\ne x_*.
\tag{32}
\end{equation}
```

Lean checks this exact counterexample for every positive $\alpha$ in
`wrong_nonnegative_fixed_point`, `wrong_fixed_point_is_not_solution`, and
`unstable_direction`. It refutes the general claim that nonnegative
backprojection weights plus clipping ensure recovery. It is an algebraic
example, not a numerical experiment with the repository's CT geometry.

(findings-specific-to-this-repository)=
## Historical code limits

### Shared grids do not make the operators adjoints

Python and browser forward projection set $a_x=(N-1)/2$ and use the same pixel centres:

$$
x_c=a_x\left(-1+\frac{2c}{N-1}\right)=c-\frac{N-1}{2}.
\tag{33}
$$

Rows use the same formula. Python's forward image sampler selects
`align_corners=True` in `grid_sample`. Historical browser backprojection scaled the 1-D pixel
centres by the rotation coefficients, broadcast them into a rotated grid
in pixel units, and interpolated the 1-D detector row at
$u/\Delta u+(M-1)/2$. That TypeScript implementation used the same
coordinates. Image and detector interpolation had zero-valued out-of-range
neighbours, retaining partial edge weights. The image sampler's normalized-grid
contract is described in
[PyTorch's `grid_sample` reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).

[archived tests/test_geometry.py](https://github.com/zhiqwang/huo/blob/37d1e524df2d3c82fcbe6942b466903eb16405ff/tests/test_geometry.py)
and
[archived packages/tests/geometry.test.mjs](https://github.com/zhiqwang/huo/blob/37d1e524df2d3c82fcbe6942b466903eb16405ff/packages/tests/geometry.test.mjs)
checked ray incidence, detector ramps, quarter-turn direction, odd/even sizes,
and shared boundary examples. The physical-unit test also checks how
physical-coordinate data converts to the current convention. These tests
establish sampled cases, not $B_\theta=A_\theta^T$ or convergence; the weights
in (13) and (14) remain different.

The Python backends expose only forward projection. PyTorch's private autograd
pair applies $A^T$ during backward and $A$ when differentiating that transpose.
`tests/test_autograd.py` checks explicit sampled matrices, finite differences,
higher derivatives, and transform compositions. JAX derives these operations
from its native sampler without a custom derivative; `tests/test_jax_reference.py`
checks numerical agreement with PyTorch. Both reconstruction helpers minimize
projection loss with Adam. Those identities do not identify
the historical uniform backfill $B_\theta$ with $A_\theta^T$.
The current jax-js backend also derives the sampled transpose natively, and
cross-language tests now compare derivatives and Adam reconstruction as well
as forward values.

### The former ART call started from zero and performed one sweep

The former TypeScript `RadonFanbeam.art` allocated a zero image, visited every
angle once in a Fisher–Yates random permutation, and returned. Each call started
from zero again. Repeated-sweep theorems describe an extension that would have
preserved the iterate between sweeps. Current Python and browser helpers run a
configured number of Adam steps on a fixed measurement loss.

### An underdetermined acquisition cannot identify every pixel image

For example, an acquisition with $512\times512$ images, $500$ detectors,
and $360$ angles has a stacked matrix

$$
\mathbf A\in\mathbb R^{180000\times262144},\qquad
\dim\ker\mathbf A\geq262144-180000=82144.
\tag{34}
$$

This is a lower bound; dependent measurements can make the nullspace larger.
Nonnegativity cannot universally restore uniqueness. For an everywhere
positive $x_*$ and any nonzero $z\in\ker\mathbf A$, let
$m=\min_j x_{*,j}>0$, $c=\max_j|z_j|>0$. Choose
$0<|\varepsilon|<m/c$. Then $x_*+\varepsilon z>0$ is distinct from $x_*$
and has exactly the same data. Uniqueness for a special image requires
additional support or structural assumptions.

### Positivity does not establish an adjoint identity

Nonnegative interpolation weights preserve nonnegative data. The positivity
checks in `tests/test_projector.py` cover that property for $A$; they do not verify

$$
\langle A_\theta x,y\rangle=\langle x,B_\theta y\rangle.
\tag{35}
$$

An informative Euclidean adjoint diagnostic uses signed inputs and reports
$|\langle Ax,y\rangle-\langle x,By\rangle|$ relative to a scale such as
$\|Ax\|\|y\|+\|x\|\|By\|$, handling a zero denominator explicitly.
If weighted inner products are intended, their symmetric positive definite
matrices must be specified in advance; the exact condition is
$H_XB=A^TH_Y$.

For a small fixed geometry, applying forward and backward projection to
basis vectors constructs the actual matrices. One can then measure transpose
mismatch, inspect $AB$, and study the ordered sweep products. A check of one
configuration does not establish a theorem for all resolutions, geometries,
clipping patterns, or schedules. An adjoint check alone also does not verify
a valid step size or convergence.

(formal-verification-in-lean)=
## Lean proofs

The checked source is
[`lean/Huo/BackwardPropagation.lean`](https://github.com/zhiqwang/huo/blob/main/lean/Huo/BackwardPropagation.lean),
imported by the default `Huo` library target. The project pins Lean and
Mathlib to **v4.24.0**, with exact dependency commits in `lake-manifest.json`.
Every theorem has a proof term; there are no `sorry`, `admit`, or custom
axiom declarations. Project warnings, including incomplete-proof warnings,
are treated as errors.

| Article result | Lean declarations | Scope |
| --- | --- | --- |
| Ray coordinate cancellation | `ray_coordinate` | Nonzero ray scale and source-to-detector distance |
| Length-model residual and contraction | `length_residual`, `length_contraction` | Scalar algebra, not the continuous integral |
| Equal-weight ART | `equal_weight_art` | Positive counts and weights imply the required nonzero denominators |
| Relaxed ART energy law | `art_energy`, `art_energy_nonincreasing` | Arbitrary real inner product space and nonzero row normal |
| Discrete residual and image error | `residual_identity`, `error_identity`, `consistent_fixed_point` | Arbitrary real linear maps, with consistency where required |
| Nullspace obstruction and ordered sweeps | `nullspace_same_data`, `nullspace_error_fixed`, `sweep_append`, `sweep_common_fixed` | Algebra of linear maps and lists |
| Nonnegative clipping | `clamp_sq_le`, `clamp_sum_sq_le`, `clamp_secant`, `clipped_error_representation` | Euclidean squared distance and diagonal slopes |
| Geometric bound and convergence | `geometric_error_bound`, `tendsto_of_error_contraction`, `linear_iterates_converge` | A supplied uniform norm-contraction hypothesis, including the affine residual recurrence |
| Bounded perturbations | `noisy_error_identity`, `perturbed_error_bound` | The additive measurement-noise term and recurrence bound (24) |
| Wrong clipped fixed point and unstable direction | `wrong_nonnegative_fixed_point`, `wrong_fixed_point_is_not_solution`, `unstable_direction` | Exact two-dimensional counterexample |

The continuous integral, its Jacobian change of variables, the Jordan-form
spectral criterion, singular-value arguments, and the identification of
PyTorch operations with the stated maps remain **paper proofs or source
analysis**. In particular, Lean does not prove that Huo satisfies (19) or
(29), or that floating-point execution recovers the true image. The
contraction hypothesis is explicit input to the verified convergence theorem,
not a conclusion smuggled in through an unproved axiom.

After installing Lean's `elan` toolchain manager, run from the repository root:

```bash
cd lean
lake exe cache get Mathlib.Analysis.SpecificLimits.Basic Mathlib.Analysis.InnerProductSpace.Basic \
  Mathlib.Tactic.FieldSimp Mathlib.Tactic.Linarith Mathlib.Tactic.NormNum Mathlib.Tactic.Ring
lake build
```

The first command fetches the Mathlib artifacts needed by the imports; the
second checks all project proofs. See the
[Lean community's project instructions](https://leanprover-community.github.io/install/project.html)
for toolchain setup. The documentation workflow also checks the Lean project
before publishing. To inspect the actual proof used for clipping, this excerpt
is included directly from the compiled source:

```{literalinclude} ../lean/Huo/BackwardPropagation.lean
:language: lean
:start-at: theorem clamp_sq_le
:end-before: /-- Euclidean squared distance
```

The mathematics establishes why residual backfill can be useful and gives
conditions for convergence of those residual-backfill models. It does not
establish convergence of the current Adam solver; that would require an
analysis of its objective, constraints and optimizer dynamics.
