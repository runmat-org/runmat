# Thermal stress

Thermomechanical analysis computes structural response caused by a temperature field, optionally in
combination with mechanical loading. RunMat treats temperature as typed state and thermal strain as
a constitutive contribution; it does not convert temperature into an arbitrary equivalent force.

The `.fea` language checks steady, transient, and nonlinear thermomechanical procedure contracts.
The numerical library admits prescribed-temperature structural realizations and a
**retained thermal state to static solid solve** path. Neither is bidirectionally
coupled. Compiled-study execution connects a thermal prefix to this same-mesh preload and a supported
buckling, prestressed modal, direct harmonic or modal-reduced harmonic consumer. This narrow graph
has local/remote numerical execution and typed publication. Explicit authored model domains and
`thermomechanical preload` preserve the thermal-to-prestress binding. The authored modal fixture
passes actual-CAD execution after pre-execution driver/worker replacement and cold declared-object
import. This does not qualify other outer-root consumer/order combinations or mid-phase restart.

## What thermomechanical equation is implemented?

For isotropic thermal expansion,

$$
\boldsymbol\varepsilon_{th}=\alpha(T-T_{ref})\mathsf I,
\qquad
\boldsymbol\sigma=\mathsf C:
(\boldsymbol\varepsilon(\mathbf u)-\boldsymbol\varepsilon_{th}).
$$

The structural equilibrium problem is

$$
\mathsf K\mathbf u=\mathbf f_{mech}+\mathbf f_{th},
\qquad
\mathbf f_{th}=\int_\Omega\mathsf B^{\mathsf T}\mathsf C
\boldsymbol\varepsilon_{th}\,d\Omega.
$$

The physical static Tet4/Tet10 branch integrates the thermal load and subtracts material-owned
thermal eigenstrain during stress/energy recovery. A prescribed field records
`OnePassPrescribedThermomechanicalProfile`; a retained solved nodal field records
`OneWaySolvedThermoelasticState`. The latter interpolates the retained temperatures at integration
points and bypasses reduced-profile stiffness/load adjustments. Other prescribed
branches include reduced beam, shell, and explicitly synthetic profiles; those do not establish
the physical solid formulation's equations or qualification. Neither path solves heat in the
structural step or feeds deformation back into temperature.

## Which thermomechanical procedures are defined?

| Procedure | Contract | Registered realization |
| --- | --- | --- |
| `thermomechanical steady` | Coupled static thermal/structural state | Prescribed profile, or retained solved temperature on Tet4/Tet10 through the Rust library |
| `thermomechanical preload` | Named retained thermal state to structural and geometric-stiffness states | Same-mesh Tet4/Tet10 static solid path; authored modal CAD replay verified, broader qualification open |
| `thermomechanical transient` | Time samples and coupled evolving state | [not yet in the current release](pending:thermomechanical.transient) |
| `thermomechanical nonlinear` | Load-factor domain and coupled nonlinear state | Same profile family with the nonlinear structural policy |

These are not bidirectional thermal solves. The transient/nonlinear coupling helpers adjust bounded
convergence policies from a thermal-strain severity measure; they do not create a thermal operator.

## How do I describe a thermally loaded structural study?

```fea
model heated_bracket_model {
    assign material.aluminum to geometry.bracket.region.body
    constraint support {
        structural.fixed { target = geometry.bracket.region.mounting_face }
    }
    load operating_temperature {
        temperature_field on geometry.bracket.region.body = 373.15 K
    }
    load_case operating { include load.operating_temperature }
}

study heated_bracket {
    geometry = geometry.bracket
    mesh = mesh.bracket
    model = model.heated_bracket_model
    analysis coupled { thermomechanical steady {} }
    output displacement = analysis.coupled.field.node.displacement
    output stress = analysis.coupled.field.element.von_mises_stress
}
```

The material needs elastic and thermal-expansion capabilities and a declared reference temperature.
Constraints control whether expansion is free, partially restrained, or fully restrained. A uniform
temperature rise in a free homogeneous body should produce expansion but no stress; that is a useful
model check.

## How do I transfer a solved temperature field into structural analysis?

Declare both domains explicitly in the study's model. Their absolute reference temperatures must
be finite and nonnegative; included models may repeat an identical declaration but conflicting
declarations reject. No domain or reference temperature is inferred from a material:

```fea
model heated_solid {
    thermal { reference_temperature = 293.15 K }
    thermomechanical { reference_temperature = 293.15 K }
    // Material assignment and boundary constraints belong here as usual.
}
```

The study graph names the exact retained thermal state and makes dependency explicit:

```fea
analysis temperature { thermal steady {} retain temperature }

analysis deformation after analysis.temperature {
    thermomechanical preload {
        temperature = analysis.temperature.state.temperature
    }
}
```

The typed state must retain mesh/topology, units, reference temperature, and producer identity.
This is a one-way preload, not a bidirectional coupled solve or an interrupted-solve restart.
The preload publishes structural-primary and geometric-stiffness states for a supported downstream
consumer. The [complete authored fixture](../examples/thermal-prestress.fea) includes
geometry, materials, constraints and a prestressed-modal consumer. Its regression replaces the
driver and worker before execution, then cold-imports only declared result objects and checks
the exact retained thermal state. It does not resume an interrupted thermal phase. The isolated
blocks above are not a standalone executable study.

The Rust numerical library provides the narrower executable path:

1. Publish a completed physical thermal result with `CanonicalSemanticState::from_thermal_result`.
2. Author a static thermomechanical consumer with a thermal `StateInput`, `Prestress` use, and a
   `Reuse` transition requiring identical topology and preserved reference temperature.
3. Call `run_linear_static_from_thermal_state` with the retained state and canonical solver mesh.

The native structural worker also accepts this retained source through schema-2
`StructuralLinearWorkload::state_input`, using the closed `ThermalPrestress` variant rather than
treating temperature as a displacement initial guess. `prepare_structural_sequence_step` binds
the source to the exact authored consumer; the prepared numerical entry retains the complete
sequence instead of erasing its thermal input. The worker independently reconstructs its resource
plan before decoding the semantic source. Estimation revision 37 adds prepared transport and
decoded-residency facts to the existing numerical thermal alignment reservations. Host preparation
and canonical profiling still materialize source bytes; this is not metadata-only admission or a
measured whole-task memory bound.

Thermal structural results retain a physical identity that commits the full solver model, source
transition and solved fields. Numerical receipts separately commit the complete transported result
and execution workload. Completion import, field recovery and prestress-state publication preserve
that physical identity. The shared same-pool scheduler retains the actual thermal publication task
dependency through the preload and supported modal/harmonic/buckling consumer. Cold numerical-prefix
replay and typed publication are supported. The authored modal fixture also passes actual-CAD
cold replay after pre-execution replacement; the full outer-root matrix, genuine durable
whole-chain qualification and remaining numerical/resource/release gates are not yet complete.

The result includes the thermal transition and a source-linked `SolverResultIdentity`. Completed
thermal results retain the producer step identity
and persistent node IDs in field order, and publication checks both against its context. Source
identity also covers retained gradient, heat-flux, storage-rate, and initial-state evidence.
Missing declared state, incompatible topology/reference, and an active prescribed temperature field that
conflicts with the retained source reject. Temperature/eigenstrain/eigenstress snapshots are
integration-point fields, not fabricated nodal fields. Recovery means deriving strain, stress,
and energy from retained solutions—not reconstructing lost data.

Thermal binding owns node alignment, material-capability validation, and source lineage. The
receiving structural assembly alone constructs equivalent loads and element recovery data.
The aligned global temperature copy is released after assembly; the immutable source and the
element-local temperatures needed for recovery are preserved. This avoids a second unused load
and recovery allocation. Static preflight additionally reserves source/alignment/load numerical
storage, local element temperatures and thermal recovery coefficients, and the 52 thermal output
values per solid element plus one coupling residual, and owned displacement/von-Mises snapshot
copies (including the temporary construction copy). These topology-derived charges participate
in memory-ceiling admission and resource-plan identity (estimation revision 25). They are not a
portable source artifact encoding/metadata bound or measured peak-memory qualification; transport
decoding and result serialization lifetimes still require separate accounting.

Completed thermal results have validated canonical encoding. Decoding checks nodal and element
inventories, temperature/gradient/flux/storage-rate histories, finite values, shapes, sample counts,
reference and initial-state evidence, and typed accepted linear/energy residual evidence. These are
consistency checks, not an independent numerical-accuracy certificate. Native `PublishState` can
consume this typed result and verifies the publication request's source identity, producer, and
topology; structural result bytes are not accepted as thermal results. Portable thermal tasks and
compiled thermal-prestress composition use these checks. They do not by themselves establish
authored source-to-source qualification or complete transport resource accounting.

`execution::ThermalWorkload` provides canonical numerical intent and a local execution entry for
an explicitly selected standalone thermal step. It retains the entire authored model and sequence
for producer lineage, binds the selected step and numerical controls in its workload identity,
and takes the canonical mesh separately. The ordinary thermal entry and workload execution share
step admission: unsupported upstream state and checkpoint/restart inputs are rejected rather than
silently initialized as a fresh solve. Portable workloads additionally reject lifecycle activations
and coupled families pending their executable contracts. Multiple thermal steps may be selected
individually without rewriting their sequence identity; this is not sequence orchestration. The
native coordinator uses separately admitted portable solve, recovery, observation and publication
tasks. Their resource plans do not establish a measured whole-process memory bound, and the
composition does not yet resume a partially completed thermal phase after driver replacement.

Thermal execution and workload preflight now share `ThermalSolveSchedule`: the sorted unique union
of grid points, active signal breakpoints, and required history coordinates, preserving left-limit
evaluation at jumps. The authored sample ceiling applies to this expanded inventory before
assembly, not just the requested snapshot count. Shared time-grid construction rejects count
overflow, nonfinite spans and indistinguishable floating-point coordinates. Preflight also counts
the four mandatory retained histories exactly as `samples * (nodes + 7 * elements)` f64 values
and rejects a resident-memory ceiling smaller than those values alone. This is a necessary memory
floor, not an upper bound: simultaneous assemblies, sparse builders, solver/coercivity workspace,
metadata, observation storage and serialization/transport still require full resource admission.

`ThermalOperatorResourceFacts` additionally derives value-independent scalar support from volume
and boundary connectivity before assembly. Every assembled conductivity/capacity and derived
system matrix must fit that support. The solver uses the existing shared coercivity-work envelope
and checks the exact prepared-system certificate against its reserved work, without assuming that
a symbolic factorization of a support superset bounds every subset's reordered fill. Preflight
reserves eight full-support CSR payloads plus diagonal damping, retained symbolic support, shared
proof workspace, and the retained field values. Shared sparse finalization counts nonzero entries
after cancellation and reserves its output buffers once with fallible allocation, avoiding geometric
growth during builder-to-CSR conversion. This covers the counted simultaneous logical matrix
payloads (including a possible accelerator prepared-system clone), not spare capacities in other
construction paths, allocator rounding, or
provider-owned storage. `ThermalWorkspaceResourceFacts` adds the audited live host vector payload:
12 nodal f64 vectors and three masks for solve/driver work; two assemblies' source, Robin,
prescribed-value and material arrays plus construction property arrays; six element f64 values
of recovery scratch. It also reuses `TopologyResourceFacts` for the two live projections. That
topology figure is the shared estimate, not a measured or exact metadata bound. Thermal conductivity,
capacity, and backward-Euler effective operators now accumulate into preplanned shared CSR buffers,
rejecting nonzero entries outside admitted support. Zero entries are compacted in place, preserving
canonical numerical identities without sparse-map builders for those operators. Essential
identity elimination now constructs its canonical rows directly, also without sparse-map builders,
including constrained diagonals absent from the original operator. Effective-pattern union counts
its actual unique entries before fallible reservation; thermal support copies use fallible index
allocation. General matrix clones and dynamic model/property/evidence metadata, observations,
and transport still require complete allocation accounting.

Thermal history observation preflight resolves the same shared field-definition inventory used
at runtime. It reserves shared stored-history/reduction costs, the larger of the monolithic and
partition evaluation peaks, and the evaluator's copied temperature/gradient/flux/storage-rate catalog
payload (`8 * (nodes + 7 * elements)`). Copy widths are derived from the shared field definitions,
not a separate three-field formula. Tet4/Tet10 tests compare this allowance with all four actual
sample buffers and exercise the exact combined ceiling. This is added under the same authored
resident ceiling before assembly. Unsupported source quantities and event histories without a
thermal event detector fail during preflight. Counts remain planner estimates/logical payloads;
dynamic plan/catalog metadata and transport are not thereby fully admitted.
Passing this partial reservation is not whole-process memory admission or GPU qualification.

Thermal history admission also reserves four owned field metadata records per sample using the
shared 4 KiB field-metadata policy from structural recovery, plus the typed schedule, time,
residual, energy-balance and observation-coordinate arrays. The exact numerical payload remains
separately reported; metadata is an estimate including field conversion scratch, not exact RSS.
The `thermal_history_memory` example uses the shared Rust counting allocator around complete
local workload execution (including its own preflight), excluding fixture and reporting-plan
construction. It fails on a measured heap-growth overrun or an insulated-temperature reference
failure. Run each case in its own process:

```sh
cargo run --release -p runmat-analysis-fea --example thermal_history_memory -- tet4 257 fields
cargo run --release -p runmat-analysis-fea --example thermal_history_memory -- tet10 4097 history
```

At revision `c9af12687` on macOS aarch64, the connected four-element Tet4 case with 257 samples measured
274,415 B additional Rust heap against a 136,943 B reservation, demonstrating the omitted
metadata. With metadata included its reservation is 4,372,303 B. The Tet10 4,097-sample case
measured 4,538,879 B against 69,353,421 B without observations, and 23,488,847 B against
1,084,098,813 B with nodal temperature history. All twelve combinations of Tet4/Tet10,
3/257/4,097 samples and fields/history passed, preserving 293.15 K to reported zero error.
This is a small spatial fixture stressed in sample count, not a spatial convergence or throughput
claim. Arbitrary model metadata, provider allocations, allocator overhead and transport remain
outside a complete producer admission guarantee.

`ThermalSourceEvidenceResourceFacts` now reserves the accumulated source-projection evidence:
one record per active heat-generation or heat-flux action per solved sample (including the
initial sample and zero-valued sources). Preflight and realization share lazy case/step selection;
membership in multiple active cases does not duplicate an action. The estimate includes owned
identifier bytes, initial/current assembly and extension-copy records, and allowances for geometric
Vec growth and minimum small-vector capacity. This joins the numerical and observation reservation
before assembly. It does not include complete authored-model ownership, realization and projection
scratch, provider memory or transport.

The memory diagnostic accepts an optional fourth argument for 0–128 uniform heat sources:

```sh
cargo run --release -p runmat-analysis-fea --example thermal_history_memory -- tet10 257 fields 128
```

Nonzero sources share a fixed total `q = rho * c`, giving the independent insulated energy-balance
reference `T(t) = 293.15 + q*t/(rho*c)` (1 K rise over 1 s). The diagnostic checks the full history,
source conservation and exact evidence cardinality, with a printed 1 mK temperature criterion
(0.1% of that rise); the no-source criterion remains 1e-8 K. The old no-source criterion failed
for heated iterative solves at errors around 4e-5 K and was not silently retained as a passing claim.
Sixteen Tet4/Tet10 × 3/257 samples × fields/history × 2/128-source cases passed on macOS aarch64.
For Tet10, 257 samples and 128 sources without observation history, 32,896 evidence records were
retained and measured Rust heap growth was 28,547,143 B against a 33,861,357 B reservation. Its
maximum temperature error was 6.22e-5 K. A separate Tet10 4,097-sample two-source history case
retained 8,194 records and measured 30,463,370 B against 1,091,239,469 B, with 3.89e-6 K error.
These remain small-spatial-fixture evidence, not general worker/native allocation guarantees.

`ThermalPreassemblyResourceFacts` composes those component estimates with authored-model
materialization under one checked resident ceiling. The shared canonical codec profiles borrowed
Serde input without constructing a value tree, encoded buffer or copied strings, reusing the
existing materialization cost policy. Its conservative byte/depth limits can reject input before
canonical encoding would; custom serializers may allocate independently. The model allowance
reserves the authored model and a possible filtered constraint copy. Unchanged thermal/modal
constraint views now borrow the original model; structural transient preparation still owns its
prepared view. This is an estimate, not an exact typed-model heap bound.

With this composition and borrowed view, the Tet10 257-sample/128-source fields case measured
28,419,674 B additional Rust heap against 36,237,881 B reserved; the Tet10 4,097-sample/two-source
history case measured 30,426,688 B against 1,092,144,217 B. Their reference errors remained
6.22e-5 K and 3.89e-6 K. The zero-source Tet4 257-sample case measured 270,327 B against
4,559,307 B and reported zero temperature error. Realization/projection scratch, provider memory,
and result materialization/transport still prevent a whole-producer admission claim.

Thermal excitation scratch is now topology-bounded: scalar projection conservation tracks unique
touched nodes in one fallibly reserved node mask, including zero and cancelling contributions.
Temperature-field conflict checks borrow excitation names from the realized inventory rather than
copying a name per selected node; only error diagnostics own those strings. The preassembly
workspace includes these masks and regional value/owner slots. Prescribed-temperature, heat-flux
and convection boundary matches stream without temporary face arrays. These changes preserve
quadrature order, regional overrides, and conflicting-temperature diagnostics; they do not bound
the separate realized-action inventory or compiled property metadata.

`ThermalRealizationResourceFacts` additionally admits every active realized action, not only
heat-generation/flux evidence producers. It streams selected authored metadata (including nested
targets and frames) and each case membership through the shared codec estimator, then adds typed
entry/case slots and the borrowed signal-pointer lookup. Expanded case metadata has the same
bounded profiling envelope and can conservatively reject a model whose authored encoding fits.
Realization reserves exact requested vector lengths, uses an in-place sorted signal lookup, and
rejects duplicate signal identifiers. Selection, breakpoint scheduling and resource preflight now
share borrowed case/action traversal; nonempty lifecycle inventories can still allocate temporary
identifier checks. Thermal execution releases realized entries immediately after assembly, so
initial and subsequent realization inventories do not coexist. This remains estimated requested
heap payload, not allocator/native/RSS coverage; signal-sampling scratch and result/transport
allocation admission still require separate qualification.

Thermal preassembly now compiles and retains one canonical property map for the run instead of
recompiling material ownership at every sample. Each assembly borrows that map, verifies its
geometry/revision/mesh binding, and updates the sample-dependent loads and boundaries. Internal
orchestration keeps the authored material model immutable; the binding check alone is not a
material-equivalence test for arbitrary same-geometry models. Property failures surface during
preassembly resource derivation, before numerical assembly. The compiler first admits estimated
lookup/assignment scratch, validates borrowed owners, and profiles their actual selected material,
section, ply and audit occurrences through the shared codec. One selection visitor serves both
profiling and owned construction, including explicit beams and homogeneous/laminate shells; there
is no separate resource-layer owner selector. The composed thermal ceiling is checked before
cloning constitutive/section/audit payloads. Unused material metadata does not multiply across
elements, and the smaller authored-workload encoding cap is not imposed on expanded occurrences.

`property_compilation_bytes` includes lookup/assignment estimates, typed map/audit/ply slots,
integrity identity vectors and shared-codec estimates for owned metadata plus one concurrent
canonical identity encoding. Sequential semantic/audit encodings and integrity rehashes are not
counted as four simultaneous buffers. Map serialization preserves ordered key/value pairs while
streaming them without a temporary entry array. These are conservative estimates, not exact
allocator or RSS limits. Explicit-topology validation and canonical identity scratch are profiled
before encoding; their peak estimate and the retained element-identity cache are admitted before
construction. Borrowed topology contracts preserve the established canonical bytes without cloning
connectivity or attachments. Selection currently traverses three times (typed validation/counting,
profiling, owned construction), reusing identities generated once for the same immutable structural
model. Canonical solver-mesh validation is also admitted before it runs in the compiler: the shared
meshing owner estimates validation indexes (including Tet10 and raw malformed-input inventories)
and canonical identity encoding. Validation indexes are released before that encoding, so those
phases use their maximum; the compiler conservatively adds it to its other phases. Profiling may
reject malformed serialization before resource admission. Thermal preassembly also checks this
estimate before operator planning derives its validated projection; the thermal entry point no
longer performs an unadmitted duplicate validation ahead of preassembly. That early gate now also
includes projection construction (cloned metadata, connectivity, lookup/neighbor indexes, coordinate
and curved-boundary quadrature scratch) and thermal sparse-support construction (row trees, growing
CSR storage and the shared coercivity graph-count tree). Raw coupling counts bound these allocations
without constructing the graphs to estimate them. Geometry validation still runs through the same
projection constructor. The final numerical reservation retains this conservative planning estimate;
it is not an exact lifetime-minimized total. Direct planner calls without an authored ceiling return
facts rather than enforce admission. Other execution paths, result transport, and whole-producer
admission remain open.

Property owner/section matching streams candidates without per-element candidate arrays, while
retaining exact coverage counts and missing/duplicate-section rejection. Semantic/audit identity
views stream solid, beam, shell, and laminate-ply sequences
without per-element temporary arrays. Integrity checking uses in-place unstable identity sorting
and releases those validation vectors before canonical rehashing. Canonical value trees and encoded
bytes still allocate and are included in the compiler's estimated materialization envelope.

Evidence for this path includes a solved-temperature-to-restrained Tet4
expansion reference with displacement, stress, reaction balance, and energy checks, plus separate
Tet4/Tet10 load and linear-temperature recovery references. Affine Tet10 quadratic-temperature
energy and an analytic curved-Tet10 map with thermal and mechanical strain are checked against
exact simplex moments. Volume averages use retained nodal temperatures, independently of the four
reported recovery samples. Curved thermal recovery compares 8-, 12-, and, when needed, 16-point
Gauss rules per Duffy coordinate and rejects unresolved integrals; this comparison is not a
certified error bound or general curved-element qualification. Spatial convergence, temperature-dependent constitutive iteration,
cross-mesh projection, portable execution/persistence, and downstream thermal-prestress procedures
are not qualified by these tests.

## Why is my thermal stress wrong?

Check coefficient-of-expansion units, reference temperature, material coverage, restraint realism,
temperature projection between meshes/regions, through-thickness gradients, contact/tie behavior,
and temperature dependence of elastic properties. Validate the free-expansion limit
$\Delta L/L=\alpha\Delta T$ and the fully restrained one-dimensional limit
$\sigma\approx-E\alpha\Delta T$ where those assumptions apply.

See [heat transfer](./thermal), [structural analysis](./structural),
[structural constraints](../constraints/structural), and [current status](../validation/status.md).
