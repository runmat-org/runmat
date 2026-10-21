# Sparse coercivity preparation and admission

The shared real-operator preparation constructs a process-local proof of coercivity
for the actual symmetric operator. Sparse LDL verifies positive inertia; the
retained electrical and homogeneous single-translation modal routes instead verify original local energies and a grounded
graph bound against every coefficient of the actual constrained operator. It retains the original
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
zero-removal change alone did not resolve the original fine electrical or modal failures.

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
The original largest electrical Tet10 case now passes through single-pass original
local-energy capture, exact source reservation (resource revision79), and shared
prepared-system reuse in signed `355b3992a`. All36 electrical tests pass, including
its unchanged field/refinement gate. The modal candidate now passes the original
damped first-mode Tet4 field/phase/power gate under its existing proof limit. No family qualification,
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

## Original modal constraint-space discriminator

The original damped rod constrains Uy and Uz at every node, and Ux at x=0.
Its exact homogeneous transform is a pure selection of one translation component;
this is not a generic unrestricted solid space with six rigid-body modes.

The shared energy projection retains eliminated source vertices as graph anchors,
checks exact bijective free-row coverage in linear time, and compares reconstructed
contributions against every coefficient of the actual reduced CSR. No identity rows
are added to the reduced operator. Independent chain eigenvalue/permutation,
malformed selection and corrupted reduced-operator tests pass. The identity-row
electrical route remains unchanged. Final164 shared algebra passes74/74.

The constraints owner recognizes only an exact homogeneous unit selection of one
translation component. Nonunit ties, prescribed offsets, multiple active axes and
extra/reference degrees of freedom do not qualify for this source. The physical
assembly captures original local stiffness blocks once, before their existing ordered
scatter; it does not repeat quadrature or change K, M, damping or mass. Bit-exact
capture comparisons pass for both element orders, all three axes and serial/parallel
assembly. Unsupported connector/coupling/mixed-element and prestressed operators
continue through ordinary shared preparation; elastic source evidence cannot certify
a geometric tangent.

Resource revision80 requires an explicit original-energy inventory in the portable
modal demand. It reserves flat source blocks, selection construction and retained
maps before execution. Missing inventory fails decoding; a self-consistent
understated plan fails native rederivation. The source moves into the existing
prepared-system owner and is consumed once. Modal solves reuse that one preparation;
they no longer copy a complete assembly summary or repeatedly construct its proof.
Proof caps, physical matrices, original meshes, references and tolerances are unchanged.

The diagnostic certifies all seven original refinements: Tet4 cells2/4/8/16 and
Tet10 cells1/2/4. The finest Tet4 operator has4624 reduced DOFs; demand13949805
operations fits18091536, and requested proof storage25606240 bytes fits56332288.
Its graph lower bound is approximately1.12938e-6; outward assembly error is below
9.75e-14. These are requested bounds, not measured RSS. This test deliberately replays
shared element blocks as a discriminator; production uses the captured originals.

Final163 original five-case damped-modal matrix passes5/5,297.05s. It covers both
orders' unchanged refinement/phase/power gates, modal/direct truncation against an
independent continuum reference, the original seven-operator discriminator, and cold
modal-state entry with recovered field phase. Finest Tet4 field errors at
k=0.5,pi/2,2 are0.00118234,0.00979734,0.00300099, below the unchanged0.02 target.
Strict FEA library/test lint162 and strict native library/test lint163 pass.

The new native both-order witness passes1/1,1.71s. It invokes the actual native
analysis kernel, rejects substituted source budgets, reopens the filesystem object
store, imports the original basis/result/receipt/publication profiles, and recovers
cold physical fields. This is kernel and cold-store evidence; it does not establish
fresh producer-process termination, measured RSS or accelerator execution for modal
extraction. Final163 native kernel regression passes85/85 with3 intentional child-entry skips
in660.46s. Final164 bridge regression passes199/199; strict bridge library/test lint
passes21.10s. These scoped results are recorded in the external completion ledger.

General solid rigid-mode proofs, prestress and other constraint spaces remain
required broader work. Historical whole-suite failures and stage status are not
replaced by this scoped pass. Architecture still reports two existing runtime renderer
module-size findings in the isolated candidate; no new waiver or debt is introduced.
