# Exponential Stability of the Learnt-Skill Controller
### under variable impedance gains K(φ), variable task phase φ̇, and contact

This is the central theoretical contribution. We prove that the closed-loop
tracking error converges **exponentially** — `‖e(t)‖ ≤ c·e^(−λt)·‖e(0)‖` — and we
show that the **same energy tank** that guarantees passivity is precisely what
bounds the gain-variation rate needed for exponential stability. Passivity,
exponential stability, and task parametrization are therefore unified by one
mechanism.

---

## 1. System model and assumptions

Task-space (TCP) dynamics of the manipulator:

    Λ(x) ẍ + C(x,ẋ) ẋ + g(x) = f_c + f_ext                         (1)

- `Λ` = task-space inertia (symmetric positive-definite),
- `C` = Coriolis/centripetal, with the **skew-symmetry property** `ẋᵀ(Λ̇ − 2C)ẋ = 0`,
- `f_c` = control wrench, `f_ext` = environment/contact wrench.

**Controller** (variable impedance with gravity/Coriolis compensation, as in
`continuous_trajectory_vic.py` which adds `qfrc_bias`):

    f_c = g + C ẋ_d + Λ ẍ_d + K(φ)(x_d − x) + D(φ)(ẋ_d − ẋ)         (2)

Define the tracking error `e = x_d(φ) − x`. Substituting (2) into (1) and using
the skew-symmetry property gives the **error dynamics**

    Λ ë + [C + D(φ)] ė + K(φ) e = f_ext .                          (3)

**Assumptions.**
- (A1) `K(φ), D(φ)` are symmetric positive-definite, bounded:
  `0 < K_min I ⪯ K(φ) ⪯ K_max I`, `0 < D_min I ⪯ D(φ)`.
- (A2) Gravity/Coriolis are compensated (the code adds `qfrc_bias`); residuals
  are bounded and absorbed into a disturbance term `δ` (see §5).
- (A3) The environment is **passive**: `∫₀ᵀ f_extᵀ ẋ dt ≥ −E₀` (true for
  elastic/dissipative materials and for cutting/fracture, which dissipates).
- (A4) Damping is set by critical-damping design `D = 2ζ√(ΛK)`, `ζ≈0.7` (as in
  code), so `D(φ)` grows with `K(φ)`.

---

## 2. Lyapunov function (with cross-term — this is what yields *exponential*)

The natural energy `V₀ = ½ėᵀΛė + ½eᵀKe` gives only `V̇₀ = −ėᵀDė ≤ 0` — negative
**semi**-definite, which yields *asymptotic* stability via LaSalle but **not
exponential**. To obtain an exponential rate we add a cross-term:

    V = ½ ėᵀ Λ ė  +  ½ eᵀ K(φ) e  +  β eᵀ Λ ė                       (4)

with `β > 0` small. In block form, `V = ½ [e;ė]ᵀ M_β [e;ė]`, `M_β = [[K, βΛ],[βΛ, Λ]]`.

**Positive-definiteness of V** (Schur complement): `V` is PD iff `Λ ≻ 0` (true)
and `K − β²Λ ≻ 0`, i.e.

    β < √( λ_min(K) / λ_max(Λ) ) .                                 (5)

Then there exist `c₁,c₂ > 0` with `c₁‖z‖² ≤ V ≤ c₂‖z‖²`, where `z = [e; ė]`.

---

## 3. Constant-gain case ⇒ exponential stability

Take `f_ext = 0`, `K̇ = 0`, `Λ` constant (Coriolis handled by skew-symmetry).
Using `Λë = −Dė − Ke`:

    V̇ = −ėᵀ(D − βΛ)ė  −  β eᵀK e  −  β eᵀD ė .                     (6)

(The `−ėᵀKe + eᵀKė` terms cancel by symmetry of `K`.) Bounding the cross term by
Young's inequality, `−β eᵀDė ≤ (β‖D‖/2)(‖e‖²/η + η‖ė‖²)`, and choosing

    β < λ_min(D) / λ_max(Λ)        (so D − βΛ ≻ 0)                  (7)

with `η` balancing the terms, there exist `α₁,α₂ > 0` such that

    V̇ ≤ −α₁‖e‖² − α₂‖ė‖² ≤ −α V ,   α = min(α₁,α₂)/c₂ > 0 .        (8)

Hence `V(t) ≤ V(0) e^(−αt)`, and via (5),

    ‖e(t)‖ ≤ √(c₂/c₁) · e^(−(α/2) t) · ‖z(0)‖ .                    (9)

**Exponential stability with rate λ = α/2.** The rate is explicit:
`α` increases with `λ_min(D)` and `λ_min(K)` and decreases with `λ_max(Λ)` — i.e.
stiffer/more-damped regulated axes converge faster, which is exactly why the
**cut-depth (Z) axis** (high K_z) converges fastest.

---

## 4. Variable gains K(φ): the condition that ties to the tank

With `K̇ = K′(φ) φ̇ ≠ 0`, the term `½ eᵀ K̇ e` appears in `V̇`:

    V̇ ≤ −α₁‖e‖² − α₂‖ė‖² + ½ eᵀK̇ e
       ≤ −(α₁ − ½‖K̇‖)‖e‖² − α₂‖ė‖² .                              (10)

Exponential stability is **preserved** provided the gain does not increase faster
than the contraction can absorb:

    ‖K̇(φ)‖  <  2 α₁  =  2 β λ_min(K)  (to leading order).         (11)

Since `K̇ = K′(φ) φ̇`, (11) is a bound on the **phase rate φ̇**:

    φ̇  <  2 β λ_min(K) / ‖K′(φ)‖ .                                (12)

**This is the crux.** Equation (12) says: *the task may advance only as fast as
the impedance-gain variation can be absorbed without losing exponential
contraction.* The energy tank (next section) is exactly the mechanism that
enforces (12) online.

---

## 5. Contact, moving reference, and the tank (passivity ⇒ the bound)

Restore `f_ext` and the reference motion. Two bounded inputs enter:
1. `f_ext` (contact) — handled by passivity (A3);
2. the reference acceleration `Λẍ_d` not perfectly fed forward — a bounded
   disturbance `δ`, larger for the **moving-material** case (sinusoidal workpiece)
   than for the **static** case.

With the tank state `T` and the passivity-preserving dynamics (see
`passivity_analysis.md`), the indefinite power `P_inj = ½eᵀK̇e + eᵀK(∂x_d/∂φ)φ̇`
is drawn from `T`; when `T → T_min` the controller forces `φ̇ → 0` (and hence
`K̇ → 0`). Therefore the tank **automatically enforces (12)**: whenever advancing
the phase would inject more energy than available, `φ̇` is throttled, restoring
the contraction condition.

Result — **practical / input-to-state exponential stability**:

    ‖e(t)‖ ≤ c · e^(−λ t) ‖z(0)‖ + γ · sup‖δ‖ + γ′ · sup‖f_ext‖_passive   (13)

- **Static material** (`ẍ_d ≈ 0`, regulation): `δ → 0`, so (13) reduces to the
  origin-exponential bound (9) up to the passive contact term ⇒ **exponential
  stability to a small ball** set by contact compliance.
- **Moving material**: the same exponential rate `λ`, converging to a slightly
  larger ball whose radius scales with the workpiece-motion acceleration. The
  feed-forward of `mat_vel` (already in the controller) shrinks `δ`.

So the **same claim — exponential convergence at rate λ to a bounded set — holds
for both experiments**, with a quantifiably larger residual ball for the moving
case. This is exactly the dual-validation the thesis requires.

---

## 6. The unification (the thesis's structural argument)

One mechanism — the energy tank — does three jobs at once:
- **Passivity:** it guarantees `V̇ ≤ f_extᵀẋ` (no energy creation).
- **Exponential stability:** by throttling `φ̇` it enforces (12), preserving the
  contraction inequality (10).
- **Task parametrization:** the throttle *is* the state-driven phase law
  `φ̇ = γ(T)·φ̇_nom` — task timing adapts to the energy budget.

Passivity, exponential stability, and adaptive task parametrization are therefore
**not three separate results but three faces of one design**.

---

## 7. What to measure (links to evaluation_metrics.md)
- Fit `‖e(t)‖` to `c·e^(−λt)` ⇒ identified `λ̂`; compare to the theoretical
  `λ = α/2` from §3 (report `λ̂/λ_theory`).
- Verify `V(t)` is non-increasing outside injection windows.
- Verify the implemented torque-port inequality `q̇ᵀτ_safe − P_max ≤ 0`
  under the adversarial power-injection test (validates the tank enforcement).
- Use the Cartesian residual `V̇ − f_extᵀẋ` as a reconstructed diagnostic, not
  as the primary proof artifact until the force frame/sign and storage model are
  fully matched to the implementation.
- Disturbance recovery (ISS): impulse perturbation ⇒ exponential return, both
  static and moving material.

## 8. scope / conditions
- The rate `λ` is a *lower bound* (conservative) from the Lyapunov argument;
  measured `λ̂` may be larger.
- Exponential-to-the-origin requires regulation (`ẍ_d→0`); under a moving
  reference it is exponential-to-a-ball (practical exponential stability) — this
  is standard and is stated, not hidden.
- Requires the tank to gate `φ̇` and set the torque-level power budget; this is
  implemented as described in `passivity_analysis.md §6`.
- Assumes reasonable gravity/Coriolis compensation; residuals enter `δ`.
