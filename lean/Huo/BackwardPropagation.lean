import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Residual backprojection

Real-arithmetic lemmas accompanying `docs/backward_propagation_proof.md`.
The linear maps are explicit hypotheses: this file does not model PyTorch,
floating-point arithmetic, continuous ray integrals, or spectral radii.
-/

namespace Huo

open Filter
open scoped Topology InnerProductSpace

section Geometry

/-- A point on the ray from `(0, source_distance)` towards `(u, -det_distance)`
maps to detector `u`. With `D = source_distance + det_distance`, the article's
ray-distance parameter satisfies `scale = t / sqrt (u²+D²)`.
-/
theorem ray_coordinate (source_distance det_distance u scale : ℝ)
    (hD : source_distance + det_distance ≠ 0) (ht : scale ≠ 0) :
    (source_distance + det_distance) * (scale * u) /
      (source_distance - (source_distance - scale * (source_distance + det_distance))) = u := by
  have h : source_distance - (source_distance - scale * (source_distance + det_distance)) =
      scale * (source_distance + det_distance) := by ring
  rw [h]
  field_simp

/-- Residual identity under the explicitly assumed single-ray length model. -/
theorem length_residual (b prediction length L : ℝ) :
    b - (prediction + length * ((b - prediction) / L)) =
      (1 - length / L) * (b - prediction) := by
  ring

/-- The strict length condition gives a scalar contraction. -/
theorem length_contraction (length L : ℝ) (hL : 0 < L)
    (hpos : 0 < length) (hupper : length < 2 * L) :
    |1 - length / L| < 1 := by
  rw [abs_lt]
  constructor
  · have h := (div_lt_iff₀ hL).2 hupper
    linarith
  · have h := div_pos hpos hL
    linarith

/-- Equal pixel weights reduce the ART coefficient to residual / total length. -/
theorem equal_weight_art (r N h : ℝ) (hN : N ≠ 0) (hh : h ≠ 0) :
    r * h / (N * h ^ 2) = r / (N * h) := by
  field_simp

end Geometry

section LinearIteration

variable {X Y : Type*} [AddCommGroup X] [Module ℝ X]
  [AddCommGroup Y] [Module ℝ Y]

/-- One angle update before nonnegative clipping. -/
def linearStep (A : X →ₗ[ℝ] Y) (B : Y →ₗ[ℝ] X) (α : ℝ) (b : Y) (x : X) : X :=
  x + α • B (b - A x)

/-- Image error propagation, `I - α BA`. -/
def errorMap (A : X →ₗ[ℝ] Y) (B : Y →ₗ[ℝ] X) (α : ℝ) : X →ₗ[ℝ] X :=
  LinearMap.id - α • (B.comp A)

theorem residual_identity (A : X →ₗ[ℝ] Y) (B : Y →ₗ[ℝ] X)
    (α : ℝ) (b : Y) (x : X) :
    b - A (linearStep A B α b x) = (b - A x) - α • A (B (b - A x)) := by
  simp only [linearStep, map_add, map_smul]
  abel

theorem error_identity (A : X →ₗ[ℝ] Y) (B : Y →ₗ[ℝ] X)
    (α : ℝ) (x xStar : X) :
    linearStep A B α (A xStar) x - xStar = errorMap A B α (x - xStar) := by
  simp only [linearStep, errorMap, LinearMap.sub_apply, LinearMap.id_apply,
    LinearMap.smul_apply, LinearMap.comp_apply, map_sub, smul_sub]
  abel

/-- Measurement error contributes an additive backprojected disturbance. -/
theorem noisy_error_identity (A : X →ₗ[ℝ] Y) (B : Y →ₗ[ℝ] X)
    (α : ℝ) (x xStar : X) (noise : Y) :
    linearStep A B α (A xStar + noise) x - xStar =
      errorMap A B α (x - xStar) + α • B noise := by
  simp only [linearStep, errorMap, LinearMap.sub_apply, LinearMap.id_apply,
    LinearMap.smul_apply, LinearMap.comp_apply, map_sub, map_add, smul_sub, smul_add]
  abel

theorem consistent_fixed_point (A : X →ₗ[ℝ] Y) (B : Y →ₗ[ℝ] X)
    (α : ℝ) (xStar : X) :
    linearStep A B α (A xStar) xStar = xStar := by
  simp [linearStep]

/-- Measurements cannot distinguish a vector from a nullspace perturbation. -/
theorem nullspace_same_data (A : X →ₗ[ℝ] Y) (x z : X) (hz : A z = 0) :
    A (x + z) = A x := by
  simp [map_add, hz]

theorem nullspace_error_fixed (A : X →ₗ[ℝ] Y) (B : Y →ₗ[ℝ] X)
    (α : ℝ) (z : X) (hz : A z = 0) : errorMap A B α z = z := by
  simp [errorMap, hz]

/-- Applying a list in order corresponds to a left fold, not a commutative sum. -/
def sweep (Ts : List (X →ₗ[ℝ] X)) (e : X) : X :=
  Ts.foldl (fun v T => T v) e

theorem sweep_append (Ts Us : List (X →ₗ[ℝ] X)) (e : X) :
    sweep (Ts ++ Us) e = sweep Us (sweep Ts e) := by
  simp [sweep, List.foldl_append]

theorem sweep_common_fixed (Ts : List (X →ₗ[ℝ] X)) (z : X)
    (hz : ∀ T ∈ Ts, T z = z) : sweep Ts z = z := by
  induction Ts with
  | nil => rfl
  | cons T Ts ih =>
      have hT := hz T (by simp)
      have hTs : ∀ U ∈ Ts, U z = z := fun U hU => hz U (by simp [hU])
      simpa [sweep, List.foldl_cons, hT] using ih hTs

end LinearIteration

section Art

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Exact relaxed ART energy identity for an arbitrary nonzero row normal. -/
theorem art_energy (a e : E) (ha : a ≠ 0) (relaxation : ℝ) :
    ‖e - (relaxation * ⟪a, e⟫_ℝ / ‖a‖ ^ 2) • a‖ ^ 2 =
      ‖e‖ ^ 2 - relaxation * (2 - relaxation) * ⟪a, e⟫_ℝ ^ 2 / ‖a‖ ^ 2 := by
  have hn : ‖a‖ ≠ 0 := norm_ne_zero_iff.mpr ha
  rw [norm_sub_sq_real]
  simp only [real_inner_smul_right, real_inner_comm e a, norm_smul,
    Real.norm_eq_abs, mul_pow, sq_abs]
  field_simp
  ring

/-- Relaxation in [0,2] cannot increase squared distance to a consistent solution. -/
theorem art_energy_nonincreasing (a e : E) (ha : a ≠ 0) (relaxation : ℝ)
    (hrelaxation : 0 ≤ relaxation) (hrelaxation2 : relaxation ≤ 2) :
    ‖e - (relaxation * ⟪a, e⟫_ℝ / ‖a‖ ^ 2) • a‖ ^ 2 ≤ ‖e‖ ^ 2 := by
  rw [art_energy a e ha relaxation]
  have h : 0 ≤ relaxation * (2 - relaxation) * ⟪a, e⟫_ℝ ^ 2 / ‖a‖ ^ 2 :=
    div_nonneg (mul_nonneg (mul_nonneg hrelaxation (sub_nonneg.mpr hrelaxation2)) (sq_nonneg _))
      (sq_nonneg _)
  linarith

end Art

section Clipping

/-- Squared scalar distance decreases when clipping toward a nonnegative target. -/
theorem clamp_sq_le (z xStar : ℝ) (hx : 0 ≤ xStar) :
    (max z 0 - xStar) ^ 2 ≤ (z - xStar) ^ 2 := by
  by_cases hz : 0 ≤ z
  · rw [max_eq_left hz]
  · rw [max_eq_right (le_of_not_ge hz)]
    have hz' : z ≤ 0 := le_of_not_ge hz
    nlinarith [mul_nonneg (neg_nonneg.mpr hz') hx, sq_nonneg z]

/-- Euclidean squared distance, expressed as a finite sum (not the Pi sup norm). -/
theorem clamp_sum_sq_le {ι : Type*} [Fintype ι] (z xStar : ι → ℝ)
    (hx : ∀ i, 0 ≤ xStar i) :
    ∑ i, (max (z i) 0 - xStar i) ^ 2 ≤ ∑ i, (z i - xStar i) ^ 2 := by
  exact Finset.sum_le_sum (fun i _ => clamp_sq_le (z i) (xStar i) (hx i))

/-- Every scalar clipping difference is a slope in [0,1] times its input difference. -/
theorem clamp_secant (a b : ℝ) :
    ∃ d : ℝ, 0 ≤ d ∧ d ≤ 1 ∧ max a 0 - max b 0 = d * (a - b) := by
  by_cases ha : 0 ≤ a
  · by_cases hb : 0 ≤ b
    · exact ⟨1, by norm_num, le_rfl, by simp [max_eq_left ha, max_eq_left hb]⟩
    · have hb' : b < 0 := lt_of_not_ge hb
      have hab : 0 < a - b := by linarith
      refine ⟨a / (a - b), div_nonneg ha hab.le, ?_, ?_⟩
      · apply (div_le_one hab).2
        linarith
      · simp [max_eq_left ha, max_eq_right hb'.le, div_mul_cancel₀ _ (ne_of_gt hab)]
  · have ha' : a < 0 := lt_of_not_ge ha
    by_cases hb : 0 ≤ b
    · have hba : 0 < b - a := by linarith
      refine ⟨b / (b - a), div_nonneg hb hba.le, ?_, ?_⟩
      · apply (div_le_one hba).2
        linarith
      · rw [max_eq_right ha'.le, max_eq_left hb]
        field_simp
        ring
    · exact ⟨0, le_rfl, by norm_num,
        by simp [max_eq_right ha'.le, max_eq_right (le_of_not_ge hb)]⟩

/-- Coordinatewise form of `e⁺ = D v`, with each diagonal entry in [0,1]. -/
theorem clipped_error_representation {ι : Type*} (xStar v : ι → ℝ)
    (hx : ∀ i, 0 ≤ xStar i) :
    ∃ d : ι → ℝ, (∀ i, 0 ≤ d i ∧ d i ≤ 1) ∧
      ∀ i, max (xStar i + v i) 0 - xStar i = d i * v i := by
  have h := fun i => clamp_secant (xStar i + v i) (xStar i)
  choose d hd using h
  refine ⟨d, fun i => ⟨(hd i).1, (hd i).2.1⟩, ?_⟩
  intro i
  simpa only [max_eq_left (hx i), add_sub_cancel_left] using (hd i).2.2

end Clipping

section Convergence

variable {E : Type*} [NormedAddCommGroup E]

/-- A uniform bound may be applied per step or to the boundaries of whole sweeps. -/
theorem geometric_error_bound (x : ℕ → E) (xStar : E) (q : ℝ) (hq : 0 ≤ q)
    (hstep : ∀ k, ‖x (k + 1) - xStar‖ ≤ q * ‖x k - xStar‖) (k : ℕ) :
    ‖x k - xStar‖ ≤ q ^ k * ‖x 0 - xStar‖ := by
  induction k with
  | zero => simp
  | succ k ih =>
      calc
        ‖x (k + 1) - xStar‖ ≤ q * ‖x k - xStar‖ := hstep k
        _ ≤ q * (q ^ k * ‖x 0 - xStar‖) := mul_le_mul_of_nonneg_left ih hq
        _ = q ^ (k + 1) * ‖x 0 - xStar‖ := by rw [pow_succ]; ring

/-- Actual convergence, not just a finite algebraic identity. -/
theorem tendsto_of_error_contraction (x : ℕ → E) (xStar : E) (q : ℝ)
    (hq : 0 ≤ q) (hq1 : q < 1)
    (hstep : ∀ k, ‖x (k + 1) - xStar‖ ≤ q * ‖x k - xStar‖) :
    Tendsto x atTop (𝓝 xStar) := by
  apply tendsto_iff_norm_sub_tendsto_zero.mpr
  apply squeeze_zero (fun k => norm_nonneg _) (geometric_error_bound x xStar q hq hstep)
  simpa using (tendsto_pow_atTop_nhds_zero_of_lt_one hq hq1).mul_const ‖x 0 - xStar‖

/-- Bounded per-step perturbations produce an error floor η / (1-q). -/
theorem perturbed_error_bound (e : ℕ → ℝ) (q η : ℝ) (hq : 0 ≤ q) (hq1 : q < 1)
    (hstep : ∀ k, e (k + 1) ≤ q * e k + η) (k : ℕ) :
    e k ≤ q ^ k * e 0 + η * (1 - q ^ k) / (1 - q) := by
  have hden : 1 - q ≠ 0 := ne_of_gt (sub_pos.mpr hq1)
  induction k with
  | zero => simp
  | succ k ih =>
      calc
        e (k + 1) ≤ q * e k + η := hstep k
        _ ≤ q * (q ^ k * e 0 + η * (1 - q ^ k) / (1 - q)) + η :=
          add_le_add_right (mul_le_mul_of_nonneg_left ih hq) η
        _ = q ^ (k + 1) * e 0 + η * (1 - q ^ (k + 1)) / (1 - q) := by
          rw [pow_succ]
          field_simp
          ring

end Convergence

section LinearConvergence

variable {X Y : Type*} [NormedAddCommGroup X] [NormedSpace ℝ X]
  [AddCommGroup Y] [Module ℝ Y]

/-- Convergence of the actual affine residual recurrence under a norm bound on I-αBA. -/
theorem linear_iterates_converge (A : X →ₗ[ℝ] Y) (B : Y →ₗ[ℝ] X)
    (α q : ℝ) (x : ℕ → X) (xStar : X) (hq : 0 ≤ q) (hq1 : q < 1)
    (hbound : ∀ e, ‖errorMap A B α e‖ ≤ q * ‖e‖)
    (hupdate : ∀ k, x (k + 1) = linearStep A B α (A xStar) (x k)) :
    Tendsto x atTop (𝓝 xStar) := by
  apply tendsto_of_error_contraction x xStar q hq hq1
  intro k
  rw [hupdate k, error_identity]
  exact hbound _

end LinearConvergence

section Counterexample

/-- The nonnegative matrix [[1,2],[2,1]] in coordinates. -/
def badBack (r : ℝ × ℝ) : ℝ × ℝ := (r.1 + 2 * r.2, 2 * r.1 + r.2)

/-- `A = I`, `b = (1,2)`, with the same residual-update-and-clamp rule. -/
def badStep (α : ℝ) (x : ℝ × ℝ) : ℝ × ℝ :=
  let r := badBack (1 - x.1, 2 - x.2)
  (max (x.1 + α * r.1) 0, max (x.2 + α * r.2) 0)

theorem wrong_nonnegative_fixed_point (α : ℝ) (hα : 0 < α) :
    badStep α (5, 0) = (5, 0) := by
  have h : α * (-6) ≤ 0 := by linarith
  norm_num [badStep, badBack, max_eq_right h]
  exact hα.le

theorem wrong_fixed_point_is_not_solution :
    (5, 0) ≠ ((1, 2) : ℝ × ℝ) := by norm_num

/-- Along `(1,-1)` the unclipped error grows by the factor `1+α`. -/
theorem unstable_direction (α : ℝ) :
    (1 - α * (badBack (1, -1)).1, -1 - α * (badBack (1, -1)).2) =
      (1 + α, -(1 + α)) := by
  norm_num [badBack]
  ring

end Counterexample

end Huo
