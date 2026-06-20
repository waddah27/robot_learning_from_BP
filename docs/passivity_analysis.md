# Passivity of Learned-Variability Impedance Control under Variable Gains and Variable Task Phase

This note gives the passivity argument for the cutting controller and states the
experiment that demonstrates it. It also records one **implementation gap** found
while writing the proof (the energy tank is currently decorative) and the fix that
makes the guarantee hold in code.

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

## 3. The two terms that break naive passivity

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

## 6. Torque-level enforcement (what the code does)

At the joint level the nominal torque `τ_nom = Jᵀf + τ_null + g` is projected onto
the passive set by a QP (`_solve_passivity_qp`):

    minimize ‖τ − τ_nom‖²   s.t.   q̇ᵀ τ ≤ P_max ,           (7)

i.e. it caps the mechanical power injected into the robot. With `P_max` tied to
the tank, `q̇ᵀτ ≤ (T − T_min)/Δt`, (7) is exactly the discrete form of the tank
constraint in §4.

### Implementation gap (found 2026-06-15) — must fix for the claim to hold
In the current code the QP uses a **constant** budget `self.tank_energy = 20`
(`bp_based_controller.py`), so `P_max = (20 − 0.001)/Δt ≈ 10⁴ W` with `Δt ≈ 2e-3`.
This bound effectively never binds. Meanwhile `update_tank_energy` integrates a
*separate* variable `self.E_t` that is **never fed back** into the QP. So the tank
is currently decorative: passivity is not actually enforced by the tank.

**Fix:** make the tank state drive the budget, and deplete/replenish it:

```python
# in __init__: self.tank = 20.0  (single source of truth, T_min, T_max)
# each step, BEFORE _solve_passivity_qp:
P_inj = 0.5 * x_tilde @ (dKdphi * phidot) * x_tilde \
        + x_tilde @ (K * dxd_dphi * phidot)          # eq. (5)
P_dis = float(kd @ vel_error**2)                      # harvested dissipation
self.tank = np.clip(self.tank + (P_dis - P_inj) * dt, self.tank_min, self.tank_max)
P_max = max(0.0, (self.tank - self.tank_min) / self.dt)   # pass into the QP
# and gate the phase:  gamma = np.tanh(self.tank / self.tank_ref); phidot *= gamma
```

With this coupling, §4–§5 hold in code, not just on paper.

## 7. Demonstration experiment (P2)

Goal: show empirically that the tank prevents instability that an un-tanked
variable-impedance controller would exhibit.

Protocol (sim, headless):
1. Run the cut with an **adversarial energy injection**: at mid-cut, command a
   transient stiffness spike `K → K_max` over a few ms (this drives `P_inj` up),
   or apply an external impulsive `f_ext` to the TCP.
2. Compare three controllers under identical perturbation:
   - **no tank** (constant high `P_max`, current code) — expect velocity/energy blow-up;
   - **tanked** (fix in §6) — expect the phase to pause, `K̇` clamp, bounded `V`;
   - **tanked + state-phase** — expect graceful slow-down and recovery.
3. Metrics: peak TCP speed, peak contact force, total stored energy `V(t)`,
   tank trajectory `T(t)`, and whether `V̇ ≤ f_extᵀẋ` holds sample-by-sample
   (plot the passivity residual `V̇ − f_extᵀẋ`; it must stay ≤ 0 for the tanked
   case and go positive for the un-tanked case).

A figure of the passivity residual staying ≤ 0 for the tanked controller, beside
it going positive (energy generation) for the un-tanked one, is the defensible
artifact for the stability chapter.

## 8. Honest scope
- The argument assumes the environment is passive (true for elastic/dissipative
  materials; cutting/fracture is dissipative, so this holds for the contact).
- `M` (Cartesian inertia) is treated as constant over a step; the discrete-time
  tank inherits the usual `O(Δt)` integration error, bounded by `T_min`.
- This is a *sim* result; sim-to-real of passivity additionally requires the
  force estimate latency/noise to be bounded (scoped as future work).
