# Canonical initial-field input qualification

This bounded Stage 5 chain uses the shared immutable `InitialFieldBinding`, not a
thermal-private artifact loader. Scalar/vector admission is generic; thermal is
the first integrated numerical consumer. It is not whole authored-study dispatch,
process restart, arbitrary topology transfer, or family qualification.

## Independent reference

The constant-property heat equation is `rho c dT/dt - div(k grad T) = q`.
On a unit cube with `k=1 W/(m K)`, `rho=2 kg/m^3`, and `c=3 J/(kg K)`,
the exact solution

```
T(x,t) = 300 + 10 x + 2 t K
grad T = [10, 0, 0] K/m
heat flux = [-10, 0, 0] W/m^2
q = 12 W/m^3
total storage rate = total applied volume power = 12 W
net prescribed-boundary power = 0 W
```

satisfies insulation on the y/z faces and prescribed temperatures `300+2t`
and `310+2t` on the x end faces. The two end-face conduction transfers cancel;
zero net exchange is not a claim that each face has zero heat transfer.

An actual steady solve first produces `T(x,0)=300+10x`. Its temperature field,
source-result identity and exact canonical mesh/node order are retained in the
input artifact. A cold canonical workload consumes that field in transient
execution with explicit boundary histories and volume generation. Tet4 and
straight Tet10 use two and four time intervals. Temperature, gradient, flux,
storage density and global power are checked against the equations above, not
only against another RunMat solve. Absolute tolerances are `1e-8` in the stated
SI units, and relative power residual is below `1e-10`; requested solver residual
is `1e-12`. This linear-space/linear-time patch checks input realization and
power accounting, not spatial or temporal convergence rates.

A separate insulated constant-temperature case checks actual producer field,
cold workload identity, all selected nodal values, zero gradient/flux/storage,
and selected-region realization, overlapping uniform/field and prescribed-temperature
conflicts, missing, duplicate, unselected and over-budget binding rejection. The capacity
checks include combined model/field input rejection before owned field preparation.

## Admission, provenance and execution boundary

### Transient numerical coordinates

Backward Euler retains `theta = T - T_reference` between steps. Constant-property
conduction annihilates the constant reference; Robin forcing uses
`h * (T_ambient - T_reference)`, and prescribed temperatures use the same shift.
Gradient, flux, storage and energy recovery operate on departures. Only published
temperature snapshots reconstruct absolute Kelvin, with prescribed values retained
exactly. This is a coordinate change, not recomputation of an input field or a change
to the heat equation, time grid, or retained physical acceptance thresholds.

If a primary solve fails the requested physical energy balance, one bounded tighter
linear solve uses the measured physical power scale and
`|sum r_i| <= sqrt(N) * ||r||_2`. Its normalization follows the shared Dirichlet-lifted
RHS (or initial residual for a zero RHS). The physical balance is recovered again;
the final acceptance gate is never rewritten. Both attempts' backend work/evidence
are counted and admitted, without raising authored resource ceilings.

An independent insulated uniform-heating regression on Tet4/Tet10 checks two steps
against `theta = q * t / (rho*c)` for increments below one absolute-temperature ulp
at 300/3000 K. Existing initial-field patches and assembly convergence tests remain
separate evidence. This correction does not qualify continuum cooling: its additional
coarse Tet4 temperature-rate and finest heat-content gates remain failed. It does not
establish process restart, accelerator execution, or whole thermal qualification.

### Immutable field inputs

The binding retains the complete existing recovered-field artifact and authored
reference. Content and topology digests, unique field/sample selection, typed
quantity, real scalar/vector storage shape, complete node count, global frame
and finite physical values must agree. Tensor[3] and Vector3 use the shared
storage-equivalence rule. Angular velocity and non-global transformations remain
closed. Admission borrows stored values; it does not recompute lost results.

The consumer uses its canonical mesh digest, not a policy-bound projection digest
or a declaration supplied by the source. Execution and numerical-intent identity
include the full binding. Thermal region and conflict rules remain authoritative.
Input materialization is checked with the borrowed combined model/field estimate
before the workload clones its field closure; numerical preassembly is checked
before assembly. Estimates are not measured whole-process memory evidence.

The native test carries a nonuniform solved producer through canonical workload
decoding, task planning, kernel input decoding and result/receipt validation on
both element orders. Native/local comparison checks transport and storage order;
the independent physics reference is the cube test above. In-process kernel
invocation is not a native process replacement or remote restart demonstration.

## Reproduce the bounded evidence

From the `runmat-analysis` workspace:

```sh
cargo test -p runmat-analysis-fea --lib execution::initial_field::tests
cargo test -p runmat-analysis-fea --lib execution::thermal_workload::tests::initial_fields
cargo test -p runmat-analysis-fea --lib thermal_schedule_admission
cargo test -p runmat-execution-runner-native --lib native_thermal_worker_consumes_source_bound_cold_initial_field
```

These focused passes do not imply unrestricted regression is green. The retained
largest electrical Tet10 and damped-modal Tet4 cases currently fail shared
coercivity proof-work ceilings before their numerical solves; they remain required.

## Remaining gates

The current authored-field draft adds explicit with-fields study/product planning
and bounded native standalone source lookup through shared analysis storage. Thermal
portable admission now uses the shared declaration validator: missing, duplicate or
extra bindings reject before numerical execution. The existing solved nonuniform
Tet4/Tet10 witness also checks authored product planning, retaining its original
temperature, gradient, flux and power references. The isolated thermal regression
set passes208/208. These checks do not establish a native authored producer-process
termination/restart envelope or measured memory qualification.

Mesh-aware source and boundary artifacts,
generated/artifact source-energy equivalence, generic authored family dispatch,
solver-specific process restart, measured residency, complete accuracy admission,
temperature-dependent closure and blocking release evidence remain required.
Other physics consumers must integrate the shared binding before claiming support.
No Stage 5 gate is promoted by this slice alone.
