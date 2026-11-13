# Thermal generated-input qualification

This Stage 5 case qualifies the shared named-callable input bridge against explicit
thermal signals and an independent heat balance. It is not full authored thermal
entry, spatial field-artifact, process-restart or family qualification.

## Reference and scope

For constant conductivity, density and heat capacity, the energy equation is
`rho c dT/dt - div(k grad T) = g`. This is the volumetric form of
[CalculiX's documented heat equation, equation 90](https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node132.html).
Here `g` has units W/m^3, rather than its documented mass-specific source `h`.
For a uniformly heated insulated body initially at uniform temperature:

```
T(t) = T_initial + g t/(rho c)
grad T = 0
heat_flux = 0
applied_power = storage_rate = g V
boundary_loss = 0
```

These independently derived values test absolute temperature, source/storage
power and zero conduction, not just generated/explicit agreement. Constant `g`
also removes temporal discretization error for backward Euler, making this an
input/constant-field patch test, not a temporal convergence demonstration.

The canonical unit tetrahedron has `V=1/6 m^3`, with `rho=2 kg/m^3`,
`c=3 J/(kg K)`, `k=1 W/(m K)`, `g=12 W/m^3` and `T_initial=300 K`.
At `t={0,0.5,1} s`, the expected uniform temperatures are `{300,301,302} K`;
applied and storage power are `2 W`. Generated `.m` functions produce source
and temperature-history samples; the initial condition samples the latter at zero.
The source excitation and initial-condition
wiring in the canonical fixture are explicit: this does not qualify authored
source-action lowering or production CAD-generated thermal meshing.

A second case removes volumetric generation and prescribes `T=300+2t K` on
every canonical boundary face using that actual compiled temperature history.
All nodes of the single Tet4/Tet10 tetrahedron are boundary nodes. The independent
uniform solution has storage `2 W` and prescribed-boundary power `-2 W` under
the solver's outward heat accounting convention, with no applied volume power
or other boundary loss. This checks nonzero generated/explicit boundary exchange;
it is not a spatial boundary-layer or unconstrained transient convergence case.

## Execution boundary

The verified focused test chain is checked `.fea` source and named project function,
ordinary compiler/callable admission, execution of actual compiled chunks,
shared canonical realization and model substitution, then governed canonical
Tet4/Tet10 thermal execution. The explicit comparator uses the same physical
model and sampling policy. Compiled requests run inline in this case; a standard
library test executable is not a native process worker. No fabricated output
vectors or private thermal history representation may substitute for execution.

The full native authored-study preparation currently admits structural families
only. This case must not disable that gate or present transfer of materialized
signals into a canonical numerical fixture as full authored thermal dispatch.

## Remaining gates

The focused native test passes for both element orders, including generated/explicit
source and prescribed-boundary cases and the independent absolute temperatures,
gradient/flux and energy balances above. The full default native library passes 159/159,
the core library passes 471/471, and the bridge library passes 91/91. Broader all-target
lint is blocked by the existing core `fea_source` integration test referencing removed
`FeaSourceCheckResult.roots`; strict core/native library lint with `occt-native` passes.
Optional CAD execution remains unverified by this slice. These tests
are not a release gate or optional CAD execution evidence.
Mesh-aware signal artifacts remain a fail-closed implementation gap. The separate
[canonical initial-field chain](thermal-field-inputs.md) integrates shared immutable
field admission into the portable thermal workload; it does not establish authored
artifact resolution, spatial source/boundary equivalence or process restart.
Nonconstant boundary/source histories,
spatial artifact equivalence, generic authored family dispatch, solver-specific
process replacement/restart, measured resource/accuracy evidence and release
qualification remain required. Radiation and temperature-dependent closure are
separate physical envelopes; existing local radiation forms do not admit a solve.
