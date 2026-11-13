# Structural affine transient — bounded continuum evidence

This Stage 4 complement isolates temporal integration, original dynamic reactions,
momentum accounting, and cold portable-object continuation from spatial initial-field
interpolation error. It does not qualify the separate sine-mode free-decay envelope
or qualify structural dynamics as a whole.

## Independent manufactured problem

The unit cube has density `rho=1 kg/m³`, Young's modulus `E=100 Pa`, Poisson ratio
zero, and unit cross-sectional area. All transverse translations are constrained;
axial displacement is fixed at `x=0`. With `A=0.01 m`, `omega=2*pi rad/s`, and
`q(t)=A*cos(omega*t)`, prescribe the initial displacement and velocity and apply:

```
u(x,t) = x*q(t)
body_force(x,t) = rho*x*(q_ddot(t) + alpha*q_dot(t))
tip_traction(t) = E*(q(t) + beta*q_dot(t))
support_reaction(t) = -tip_traction(t)
whole_body_inertia(t) = rho*q_ddot(t)/2
whole_body_mass_damping(t) = rho*alpha*q_dot(t)/2
applied_resultant(t) = tip_traction(t) + body_force(1,t)/2
```

These expressions follow directly from one-dimensional momentum balance embedded
in the three-dimensional solid. They do not use RunMat matrices or an eigensolve.
Stiffness-proportional damping enters the tip/support traction, while the global
internal force resultant cancels. Cases are undamped and Rayleigh damping
`alpha=0.8 s^-1`, `beta=0.002 s`. The affine spatial field is representable by both
straight Tet4 and Tet10; integration ends at `0.2 s`.

## Mandatory checks

The existing field/work test retains independent displacement, velocity,
acceleration, strain energy, kinetic energy, damping dissipation, and external
work. The added history test evaluates every retained sample against independent
support reaction, inertia, and mass-damping resultants. All nonzero temporal RMS
errors must decrease by more than three for intervals `20/40/80` and finish below
`0.0002`, using fixed physical normalization. Undamped damping must remain zero.

Original applied resultants are checked directly, internal elastic resultants
must cancel, and retained reaction fields must have exactly three values per node.
Only supported axial nodes carry axial reactions; the whole-body transverse
reaction resultant must vanish. Individual transverse constraint reactions are
not required to vanish. Original dynamic momentum defect, zero support work,
and mechanical energy closure remain mandatory checks.

At 80 intervals, two temporal blocks round-trip the original canonical workload,
mesh, requests, checkpoint, segment samples, assembled result, and numerical
receipt. Recovery is admitted against the original objects, retains the independent
continuum checks, and compares retained histories with uninterrupted execution.
Foreign initial conditions or topology cannot reuse that recovery admission.
This is portable-object continuation, not a failed-process or remote-worker test.

## Verification and completion boundary

The focused three-test group passes for both orders and both damping regimes
(2026-10-02, session10973, 73.18 s). Finest reaction RMS errors are approximately
`8.73e-6`–`9.23e-6`, with approximately fourfold temporal error reduction.
The current-source rerun (session74492, 98.98 s, three tests passing) also verifies
exact parity of every retained physical snapshot channel and scalar residual
history. This remains a focused result, not an unrestricted FEA suite pass.

The original free-decay spatial/temporal/cold failures in
[solid-rod-transient.md](solid-rod-transient.md) remain unchanged and mandatory.
Finer cubic-mesh free-decay measurements are separate, not substitutes for this
affine problem or retroactive passes for the failed coarse meshes. Prescribed
motion/work, general bending/shear/Poisson response, generated-input equivalence,
study-accuracy acceptance, measured resources, actual process/remote restart,
modal/direct agreement, and blocking release evidence remain open gates.

```sh
cargo test --locked -p runmat-analysis-fea --lib \
  pipeline::transient::rod_qualification::affine_forcing -- --nocapture --test-threads=1
```
