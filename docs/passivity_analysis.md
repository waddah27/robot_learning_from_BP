# Passivity of Learned-Variability Impedance Control under Variable Gains and Variable Task Phase

This note gives the passivity argument for the cutting controller, documents the
implementation used in `mjModeling/controllers/vic_controller`, and states the
experiment that demonstrates it.

## 1. System and port

Let the tool (scalpel TCP) Cartesian state be position `x ∈ ℝ³`, velocity `ẋ`.
The reference comes from the behaviour prior, parametrized by a phase variable
`φ ∈ [0,1]`, advancing at a state-modulated rate `φ̇ = γ · φ̇_nom` (see §5).

The impedance control law applied at the TCP is

    f = K(φ)·(x_d(φ) − x) + D(φ)·(ẋ_d(φ) − ẋ)              (1)

with diagonal, time-varying, positive-definite gains `K(φ), D(φ) > 0` produced by
the learned-variability QP. The environment (material) acts through an external
force `f_ext`. We analyse passivity of the map

    u = f_ext   ↦   y = ẋ            (port (f_ext, ẋ)).

A system is **passive** w.r.t. this port if there exists a storage function
`V ≥ 0` with

    V̇ ≤ yᵀu = f_extᵀ ẋ .                                   (2)

Physically: the controller cannot create energy; it can only store or dissipate
what the environment injects. This is the property that guarantees stable contact
with an unknown passive environment.

## 2. Storage function

Define the position error `x̃ = x_d(φ) − x` and use

    V = ½ ẋᵀ M ẋ  +  ½ x̃ᵀ K(φ) x̃  +  T                     (3)

(kinetic + impedance potential + tank energy `T ≥ 0`). `M` is the Cartesian
inertia.

## 3. The two terms that break the nominal passivity balance

Differentiating (3) along the closed loop produces, besides the desired
`f_extᵀẋ` and the dissipation `−ẋ̃ᵀ D ẋ̃ ≤ 0`, two problematic terms:

1. **Variable-gain term** `½ x̃ᵀ K̇(φ) x̃`. When stiffness *increases*
   (`K̇ > 0`) the stored potential rises without any input — an energy source.
2. **Variable-phase / reference term** `x̃ᵀ K ẋ_d = x̃ᵀ K (∂x_d/∂φ) φ̇`. A moving
   reference injects power into the spring.

Both are bounded but sign-indefinite, so (2) is **not** automatically satisfied.
This is the standard obstacle for variable impedance and for time-scaled motion.

## 4. Energy tank closes the gap

Introduce a tank with state `T` and dynamics

    Ṫ = ẋ̃ᵀ D ẋ̃  −  P_inj ,    T ≥ T_min > 0                 (4)

where `ẋ̃ᵀ D ẋ̃ ≥ 0` is the (harvested) dissipated power and `P_inj` is the power
drawn by the two problematic terms,

    P_inj = ½ x̃ᵀ K̇ x̃  +  x̃ᵀ K (∂x_d/∂φ) φ̇ .               (5)

Substituting (4)–(5) into the derivative of (3) cancels the indefinite terms
exactly, leaving

    V̇ = f_extᵀ ẋ  −  ẋ̃ᵀ D ẋ̃  +  (Ṫ + P_inj − ẋ̃ᵀ D ẋ̃)
       = f_extᵀ ẋ  −  ẋ̃ᵀ D ẋ̃  ≤  f_extᵀ ẋ .                  (6)

Hence (2) holds **as long as the tank can supply `P_inj` without violating
`T ≥ T_min`.** The controller enforces this by allowing the gain increase / phase
advance only while `T > T_min`; when the tank is depleted, `K̇` is clamped to ≤ 0
and `φ̇ → 0` (the phase pauses). This makes passivity **structural**, independent
of the gains the learned-variability QP chooses and independent of the phase rate.

## 5. Coupling to the learned-variability and state-phase contributions

- **Variable gains** `K(φ)` from the precision profile change only at the rate the
  phase advances, so `K̇ = (∂K/∂φ) φ̇`. Both indefinite terms in (5) are therefore
  proportional to `φ̇`; gating `φ̇` on the tank gates *both* at once.
- **State-driven phase** `φ̇ = γ(T,·) φ̇_nom` with `0 ≤ γ ≤ 1` (the tanh/sigmoid
  modulation): since `γ ≤ 1`, the injected power is bounded above by the nominal
  case, and `γ → 0` as `T → T_min`. The state-phase law is thus passivity-
  preserving by construction.

This is the thesis's theoretical claim: *learned-variability gains and
state-driven task timing are admissible because the tank renders the closed loop
passive for any `K(φ) > 0` and any `0 ≤ γ ≤ 1`.*

## 6. Torque-level enforcement in code

At the joint level the nominal torque `τ_nom = Jᵀf + τ_null + g` is projected onto
the passive set by a QP (`_solve_passivity_qp`):

    minimize ‖τ − τ_nom‖²   s.t.   q̇ᵀ τ ≤ P_max ,           (7)

i.e. it caps the mechanical power injected into the robot. With `P_max` tied to
the tank, `q̇ᵀτ ≤ (T − T_min)/Δt`, (7) is exactly the discrete form of the tank
constraint in §4.

The implementation now uses one live tank state:

```python
T[k+1] = clip(T[k] + (P_dis - P_inj) * Δt, T_min, T_max)
P_max  = max(0, (T[k+1] - T_min) / Δt)
γ      = tanh((T[k+1] - T_min) / T_ref)
φ̇      = γ φ̇_nom
```

where:

- `P_dis = ẋ̃ᵀDẋ̃` is harvested damping power.
- `P_inj = P_gain + P_ref` is the positive power required by stiffness increases
  and reference motion.
- `_solve_passivity_qp` enforces `q̇ᵀτ ≤ P_max`. If the numerical QP fails, the
  code falls back to the closed-form Euclidean projection onto the same halfspace.
- `E_t` is retained only as a compatibility alias for plotting; `tank_energy` is
  the single source of truth used by the QP.

The controller logs `tank`, `tank_gamma`, `P_dis`, `P_inj`,
`tank_balance_residual`, `power_nominal`, `power_safe`, and `power_limit` at
every timestep. The experiment harness reports `tank_min_J`, `tank_gamma_min`,
`power_violation_frac`, and `max_power_margin_W`; for the torque-level implementation,
`power_violation_frac` should be zero up to numerical tolerance.

### Implementation note
This enforcement is a discrete-time, torque-level safety projection. It supports
the passivity claim at the implemented port `q̇ᵀτ`; the Cartesian residual
`V̇ − f_extᵀẋ` remains an empirical diagnostic because it is reconstructed from
logged finite differences and an approximate effective Cartesian mass.

## 7. Demonstration experiment (P2)

Goal: show empirically that the implemented tank prevents commanded positive
joint-power injection that an un-tanked controller would allow.

Protocol (sim, headless):
1. Run the cut with a **positive joint-power perturbation**: during the middle of
   the cut, add a known positive-power torque component before `_solve_passivity_qp`.
   This is a validation hook only; it is disabled in normal experiments.
2. Compare three controllers under identical perturbation:
   - **no tank**: no torque projection, no phase gate;
   - **tank projection**: live tank drives the torque-power QP, no phase gate;
   - **tank + phase gate**: torque-power QP plus tank-modulated phase speed.
3. Plot the implemented inequality directly:

       q̇ᵀτ_safe − P_max ≤ 0

   together with `T(t)`, `P_max`, `q̇ᵀτ_nominal`, `q̇ᵀτ_safe`, and phase progress.

Current validation artifact:

    python experiments_eval/passivity_adversarial.py

produces:

    results/figures/passivity_adversarial_cork.png

Observed result on cork:

| controller | power-budget violation | max margin |
|---|---:|---:|
| no tank | 12.0% | +5876 W |
| tank projection | 0.0% | 0 W |
| tank + phase gate | 0.0% | 0 W |

This is the defensible passivity figure: the un-tanked controller exceeds the
counterfactual energy budget during the injected power burst, while the tanked
controllers keep `q̇ᵀτ_safe` below `P_max` throughout the run.

## 8. Assumptions
- The argument assumes the environment is passive (true for elastic/dissipative
  materials; cutting/fracture is dissipative, so this holds for the contact).
- `M` (Cartesian inertia) is treated as constant over a step; the discrete-time
  tank inherits the usual `O(Δt)` integration error, bounded by `T_min`.
- This is a *sim* result; sim-to-real of passivity additionally requires the
  force estimate latency/noise to be bounded (scoped as future work).
