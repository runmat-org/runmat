# Native thermal radiation and original temperature histories

Status: the original native steady and temporal witness matrix passes for both element
orders. This document supplements
[the retained radiation formulation](thermal-radiation.md); it does not complete Stage 5.

## Execution and admission

The original gray surface-to-ambient law, constant-property Tet4/Tet10 conductivity/capacity,
damped Newton and backward Euler run through the normal local thermal study task pool.
The numerical task returns primary fields; shared observation tasks evaluate the authored requests.
An opaque primary-field preparation validates the original selectors and reserves complete numerical,
source, history, metadata and output storage. It cannot execute inline observations. The full inline
API retains its combined reservation. Resource model revision 91 distinguishes the phase composition
and bounds series output by the typed statistical operation and extrema tie policy. Mean/RMS/variance
do not reserve extrema records; extrema and envelope operations reserve their actual location inventory.
Complete retained contributions remain covered by the shared input and encoding budgets.

The original 8 GiB resident ceiling, sample cap, meshes, residual target and Newton/trial limits
are unchanged. The witness explicitly configures worker capacity equal to its authored envelope;
task hints do not resize production pools. Reservations are not measured process RSS.

The coordinator validates borrowed authored intent with original initial-field bindings. It does
not compare that intent against a fieldless reconstruction. Initial-field resources are checked
before numerical ownership copies the bindings.

## Independent scientific evidence

Steady unit-slab conduction/radiation uses the retained independent 64-step monotone scalar
bisection reference. Both element orders check temperature, gradient, flux and prescribed/radiated
power. Cooling uses the original continuum `T(x,t)=400-10x-20t²`, original manufactured sources,
and 8/16/32 intervals. All four original temperature/gradient/flux/storage refinement criteria
remain mandatory, including refinement greater than 1.8 and finest temperature error below 0.1 K.
Source tables are exact at solve knots; they do not certify interpolation between knots.

The full native `--thermal-radiation-only` matrix exits successfully on revision 91. Peak errors
are shown in the original channel order; gradient, flux and storage errors are normalized as in the
original continuum tests. Every channel improves by more than 1.8 at each refinement.

| Element | Intervals | Temperature (K) | Gradient | Heat flux | Storage power |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tet4 | 8 | 0.22559350 | 0.04511870 | 0.04511870 | 0.06242210 |
| Tet4 | 16 | 0.11278010 | 0.02255602 | 0.02255602 | 0.03121887 |
| Tet4 | 32 | 0.05638247 | 0.01127649 | 0.01127649 | 0.01561164 |
| Tet10 | 8 | 0.21729540 | 0.04343625 | 0.04343625 | 0.06233009 |
| Tet10 | 16 | 0.10864631 | 0.02171771 | 0.02171771 | 0.03118164 |
| Tet10 | 32 | 0.05431748 | 0.01085763 | 0.01085763 | 0.01559522 |

The witness starts from a compiled-domain fixture through the public study execution route.
It does not establish `.fea` parser or native CAD coverage. The finest-only diagnostic selector
is not a substitute for the mandatory full temporal matrix.

A real native linear-conduction producer supplies the original spatial initial temperature.
Producing workers terminate before cold closure import. The exact original field artifact enters
the subsequent transient workload through the shared recovery and initial-field binding owners.

## Cold closure and acceptance

Cold import reads only the declared immutable closure. It compares complete original fields,
histories, nonlinear acceptance, numerical receipt, accuracy, semantic states, observations and
resource-accounting transcript, then republishes the same root. It does not rerun equilibrium or
reconstruct lost fields. Thermal receipt schema 2 now carries the shared solver's actual relative residual in its
prepared equation space. Steady, transient and nonlinear correction reporting no longer divides
by a load norm floored at one watt. The solver supplies its original right-hand-side scale, or
initial-residual scale for a homogeneous warm solve; balanced zero equations retain zero error.
Missing or nonfinite evidence rejects. Legacy receipt schema/prefix 1 rejects rather than acquiring
the new interpretation. Numerical intent binds this reporting method. Solver stopping and physical
acceptance are unchanged. Three independent scaling/zero-equation/evidence regressions include
public both-order steady conduction and cold receipt roundtrip; all 212 thermal, 206 bridge and
9 native thermal transport tests pass, with strict package lint. The earlier full native radiation
matrix is historical revision-91 evidence before this receipt change; the current original core
thermal matrix and native transport checks validate the updated reporting. It is not a fresh
whole native process-matrix rerun.

Accepted equilibrium, increment and physical energy residuals retain
1e-9 targets and the original correction/backtracking limits.

The scientific matrix checks representative transport-format corruption, root and final inventory
omission/substitution, foreign authority/scope/encryption, reordered/omitted/extra phases, altered
workload/response/accounting and foreign full material context. The existing generic thermal
lifecycle keeps its exhaustive missing/corrupt check for every closure object. Full positive closure
validation remains present in both helpers; provenance hashes are not mathematical proof.

Missing mandatory spatial convergence remains indeterminate; the result publishes, marked
indeterminate, only under a `markIndeterminate` unavailable-evidence policy and is otherwise blocked.
These native temporal references do not supply arbitrary spatial, curved-domain, temperature-dependent
material, failed-process numerical restart, remote/distributed/GPU or release qualification.

## Reproduce

From the repository, using the existing configured Cargo target:

```sh
cargo test -p runmat-analysis-fea --lib pipeline::thermal::physical::nonlinear
cargo test -p runmat-analysis-fea --lib tests::thermal_schedule_admission
cargo test -p runmat-execution-runner-native --test analysis_process_conformance -- --thermal-radiation-only
```
