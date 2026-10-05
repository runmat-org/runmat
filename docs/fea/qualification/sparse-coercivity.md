# Sparse coercivity preparation and admission

The shared real-operator preparation constructs a process-local sparse LDL proof
of positive inertia for the actual symmetric operator. It retains the original
source pattern and numerical-value identities, complete dimension, positive scaled
pivot evidence and exact factor/update counts. Serialized metadata cannot replace
that numerical construction.

## Numerical support

Proof construction omits only coefficients exactly equal to zero from its graph.
It preserves every nonzero coefficient, including subnormal and very small values;
there is no drop tolerance, operator modification or new physical discretization.
Original symmetry checks still visit the source entries. The established scaled
positive-pivot threshold and reverse Cuthill–McKee ordering remain unchanged.
Essential-constraint preparation already removes zeros on its linear path; this
change does not resolve the original fine electrical or modal failures.

Tests cover an independently SPD tridiagonal matrix stored in full CSR, exact
source preservation, tiny positive/negative couplings, changed-value rejection,
and singular, indefinite and asymmetric operators. An actual strictly diagonally
dominant five-row SPD matrix demonstrates that zero removal can change the order
and increase updates from10 for the structural upper graph to11 for the numerical
subgraph. An upper graph's own successful elimination is therefore not a bound
for every operator eventually assembled from that support.

## Planning and execution

Resource revision71 makes modal extraction, buckling and response-spectrum
missing-mass work reserve the existing shared `CoercivityWorkEnvelope`. These
planners no longer perform a factorization of an upper graph to estimate the
actual proof workspace. The envelope retains the original16x factor-fill and
1024x update limits, checked arithmetic and logical construction-storage bound.
Modal failure diagnostics distinguish admission failure from a noncoercivity claim.
Buckling retains typed construction, real-operator admission and preconditioner
errors, including the original proof-resource cause. Authored task ceilings remain unchanged: a complete demand that exceeds its
ceiling must reject before execution. Direct static and transient paths already
use this shared envelope.

The source CSR pattern defines the proof ceiling even when its exact numerical
support is smaller. Actual certificate counts are verified against source-bound
admission. This is a logical resource bound, not measured peak resident memory.
The original largest electrical Tet10 and damped first-mode Tet4 cases still
reject under their existing proof limits. No field-accuracy, family maturity,
accelerator, distributed or release gate is promoted.

## Reproduction

From the analysis workspace:

```sh
cargo test -p runmat-analysis-fea --lib algebra::
cargo test -p runmat-analysis-fea --lib planning::resource::
cargo test -p runmat-execution-runner-native --lib analysis_thermal::tests::prestress::
cargo test -p runmat-execution-runner-native --lib analysis_kernel::response_spectrum_tests::
```

The original failure cases remain mandatory:

```sh
cargo test -p runmat-analysis-fea --lib harmonic_quadratic_potential_has_tet10_exact_field_and_tet4_convergence
cargo test -p runmat-analysis-fea --lib damped_first_modal_tet4_field_phase_and_power_converge_to_analytic_mode
```
