# Procedure Sequences, Observations, and Accuracy

An FEA model describes physical objects. An analysis sequence describes when those objects are
active, which procedure runs, what state crosses a procedure boundary, which results must be
computed, and what evidence makes the result acceptable.

These are separate contracts in RunMat:

| Contract | Owns | Does not own |
| --- | --- | --- |
| `fea.AnalysisSequence` | Procedure order, dependencies, transitions, and branches | Worker placement or solver task scheduling |
| `fea.ProcedureStep` | Physics procedure, independent domain, activations, state, observations, accuracy, and restart | Materials, loads, or geometry definitions |
| `fea.Observation` | Scientific quantity, target, recovery, sampling, reduction, and retention | Plot styling or figure creation |
| `fea.AccuracyPlan` | Required numerical evidence, tolerances, resource ceilings, and limit action | Formulation maturity or publication status |

The objects are immutable and portable. A sequence authored in RunMat code serializes into the
same solver-neutral analysis contract used by a `.fea` document, a local process, or a cluster
job. Execution placement may change without changing the sequence identity or physical meaning.

## Complete Static Sequence

The following extends a model whose geometry, material, fixed constraint, force signal, and force
excitation have already been created. The observation asks for displacement; `plot` is not
involved in selecting or recovering that field.

```matlab
displacement = fea.Observation.field( ...
    "tip_displacement", ...
    fea.ObservationQuantity.displacement(), ...
    fea.ObservationTarget.boundary(loadPad), ...
    fea.FieldLocation.node(), ...
    fea.FieldLocation.node(), ...
    fea.RecoveryPolicy.native(), ...
    fea.ObservationComponent.all(), ...
    fea.CoordinateFrame.globalFrame(), ...
    fea.SignConvention.native(), ...
    fea.ComplexRepresentation.real(), ...
    fea.SamplingPolicy.finalSample(), ...
    fea.ReductionPolicy.none(), ...
    fea.ObservationStorage.new( ...
        fea.SignalPrecision.real64(), ...
        fea.CompressionPolicy.lossless(), ...
        fea.RetentionPolicy.canonical(), ...
        fea.PartialAccessPolicy.manifestAndChunks()), ...
    fea.ObservationPurpose.visualization(), ...
    fea.ObservationCost.new(25000, 400000, 24000, 24000, 0));

displacementAccuracy = fea.AccuracyCriterion.new( ...
    "tip_displacement_spatial", ...
    fea.AccuracyTarget.observation("tip_displacement"), ...
    fea.ErrorContribution.spatialDiscretization(), ...
    fea.AccuracyTolerance.relative(0.01), ...
    fea.RequiredAccuracyEvidence.convergenceStudy(), ...
    true);

accuracy = fea.AccuracyPlan.new( ...
    "static_acceptance", ...
    {displacementAccuracy}, ...
    fea.UnavailableEvidencePolicy.fail(), ...
    fea.AccuracyLimitAction.refine(), ...
    fea.NumericalEquivalencePolicy.relativeTolerance(1e-12), ...
    fea.AccuracyResourceCeiling.new(3, 2, 100, 600000, 2000000000));

fixedActivation = fea.StepObjectActivation.new( ...
    fea.StepObjectRef.constraint("fixed_base"), ...
    fea.ActivationLifecycle.activate());
forceActivation = fea.StepObjectActivation.new( ...
    fea.StepObjectRef.excitation("tip_force"), ...
    fea.ActivationLifecycle.activate());

equilibriumState = fea.StateOutput.new( ...
    "static_equilibrium", ...
    fea.StateKind.structuralPrimary(), ...
    fea.StateCompleteness.complete(), ...
    fea.StatePersistence.canonical());

staticStep = fea.ProcedureStep.configure( ...
    fea.StructuralStep.static("static"), ...
    fea.IndependentDomain.staticDomain(), ...
    {fixedActivation, forceActivation}, ...
    cell(0, 0), ...
    {equilibriumState}, ...
    {displacement}, ...
    accuracy, ...
    fea.RestartPolicy.fresh());

sequence = fea.AnalysisSequence.new( ...
    "bracket_sequence", {staticStep}, cell(0, 0), cell(0, 0), cell(0, 0));

model = fea.model("bracket", geom, ...
    "Template", "linear_static_structural", ...
    "Materials", {steel}, ...
    "Constraints", {fixed}, ...
    "Signals", {forceSignal}, ...
    "Excitations", {tipLoad}, ...
    "ExcitationCases", {serviceLoads}, ...
    "Sequence", sequence);
```

`fea.model` accepts either `Sequence` or `Steps`, never both. `Steps` is convenient for independent
single-step models. Use `Sequence` whenever activation, state transfer, branching, or governed
restart matters.

## Independent Domains

Every procedure step has one typed independent domain:

| Constructor | Meaning |
| --- | --- |
| `fea.IndependentDomain.staticDomain()` | One static equilibrium state |
| `.timeSeconds(origin, start, end)` | Physical time interval with an explicit total-time origin |
| `.frequencyHertz(values)` | Explicit finite frequency samples |
| `.loadFactor(start, end)` | Static or nonlinear load-factor interval |
| `.eigenvalue(count)` | Requested eigenpair count |
| `.studyParameter(id, values)` | Explicit samples of a named study parameter |

Time, frequency, load factor, eigenvalue, and a study parameter are not interchangeable numeric
axes. A transition must be meaningful for both the producer and consumer domains.

## Step-Scoped Object Lifecycles

An activation binds an authored model object to one step. Construct the reference with
`fea.StepObjectRef.constraint`, `.boundary`, `.connector`, `.interface`, `.excitation`,
`.initialCondition`, or `.materialParameter`. A material-parameter id has the form
`material_id.parameter_name`.

| Constructor | Meaning |
| --- | --- |
| `fea.ActivationLifecycle.activate()` | The object becomes active in this step. |
| `.continueFrom(stepId)` | The same object remains active from an upstream step. |
| `.modify(priorStepId, priorObject)` | A new authored object replaces an upstream object. |
| `.deactivate(stepId)` | An upstream object becomes inactive in this step. |

Absence is not continuation. Validation rejects duplicate activations, unknown object ids,
continuation from a non-ancestor, modification without a new identity, and continuation of an
object that was not active upstream.

## State Transfer and Branching

`fea.StateOutput.new(id, kind, completeness, persistence)` declares immutable state published by a
step. State kinds include structural primary and kinematic state, stress/strain and material or
contact history, geometric stiffness, thermal and fluid state, static or complex electromagnetic
state, complex acoustic state, modal bases, moving meshes, coupled-interface state, and integrator
state.

A consumer declares either initial state or an upstream state:

```matlab
initial = fea.StateInput.initialConditions( ...
    "initial_temperature", fea.StateKind.thermal(), ...
    fea.StateUse.physicalInitialState(), ...
    fea.InitialStateConflictPolicy.reject());

prestress = fea.StateInput.upstream( ...
    "prestress", fea.StateKind.structuralStressStrain(), ...
    "preload", "converged_stress", fea.StateUse.prestress(), ...
    fea.InitialStateConflictPolicy.reject());
```

The conflict policy may reject, replace an initial condition, or augment disjoint quantities. It
cannot silently merge two definitions of the same state.

Use `fea.SequenceDependency.new(predecessor, successor, reason)` for semantic ordering. Reasons
are `stateTransfer`, `authoredOrdering`, `sharedInterface`, and `adaptation`. A transfer is
described separately:

```matlab
transition = fea.StateTransition.new( ...
    "prestress_transfer", "preload", "modes", "converged_stress", ...
    fea.StateTransitionMethod.linearize(), ...
    fea.TopologyTransitionPolicy.requireIdentical(), ...
    fea.ReferenceStatePolicy.preserve());
```

Other transition methods are reuse, interpolation, conservative projection, deriving an operator
contribution, and initial-guess-only transfer. Topology may require identity, an evidenced
projection, or an adapted-mesh lineage. Reference state may be preserved, reset to the receiving
step origin, or supplied as an explicit linearization point.

`fea.SequenceBranch.new(id, producerStep, stateIds, consumerSteps)` publishes named states once and
permits independent consumers to proceed. Branches describe physical dependencies; the execution
planner decides whether available local or cluster resources can run those consumers concurrently.

## Restart Policy

| Constructor | Behavior |
| --- | --- |
| `fea.RestartPolicy.fresh()` | Start from the declared state inputs. |
| `.continueFromCompletedState(stepId, stateId)` | Continue from a complete upstream state. |
| `.resumeCheckpoint(checkpointId, requiredStateDigest)` | Resume only the digest-fenced checkpoint. |

A checkpoint id alone is insufficient. The required state digest prevents a checkpoint from a
different model, sequence, topology, or state from being accepted accidentally.

## Observation Requests

The four scientific observation forms share one explicit signature:

```text
fea.Observation.field|history|probe|reduction(
    id, quantity, target, sourceLocation, outputLocation, recovery,
    component, frame, sign, complexRepresentation, sampling,
    reduction, storage, purpose, cost)
```

- `field` retains a spatial field.
- `history` retains values over the independent domain.
- `probe` samples a point, path, connector, constraint, port, interface, or spatial target.
- `reduction` evaluates a governed spatial or series reduction.

Targets are constructed with `fea.ObservationTarget.boundary(region)`, `.volume(region)`,
`.point(position)`, `.path(positions)`, `.interface(id)`, `.port(id)`, `.connector(id)`,
`.constraint(id)`, or `.wholeModel()`. Region targets retain geometry identity during code
authoring; aliases in a `.fea` document resolve to the same canonical region ids. Meshing rewrites
boundary references to persistent source-topology ids for constraints, boundaries, interfaces,
excitations, initial conditions, and observations together.

Quantities cover structural motion, strain, stress, reactions and energy; thermal temperature,
gradient, flux and rate; electric and magnetic fields and sources; acoustic pressure, particle
velocity and sound-pressure level; fluid pressure, velocity, vorticity, mass flow and wall shear;
modal values; and residual, balance, constraint, and contact diagnostics. The quantity fixes the
canonical scalar, vector, or tensor shape.

Source and output locations are typed `fea.FieldLocation` values. Recovery is selected explicitly
from native values, extrapolation, volume- or area-weighted averaging, conservative projection, or
shape-function interpolation. Component, coordinate frame, sign convention, and real/imaginary/
magnitude/phase/complex representation are also explicit. Unsupported combinations fail model
validation instead of being approximated silently.

A solid path observation is an ordered set of physical positions, not a list of nearby mesh nodes.
Use a nodal source, `fea.FieldLocation.point()` output, and
`fea.RecoveryPolicy.shapeFunctionInterpolation()`. RunMat locates each position in its owning Tet4
or Tet10 element, evaluates that formulation's shape functions, and retains the ordered physical
position and owning element with the result. A point outside the admitted solid fails resolution.
For example:

```matlab
path = fea.ObservationTarget.path([ ...
    0.00, 0.00, 0.00;
    0.01, 0.00, 0.00;
    0.02, 0.00, 0.00]);

temperature_path = fea.Observation.field( ...
    "temperature_along_centerline", ...
    fea.ObservationQuantity.temperature(), ...
    path, ...
    fea.FieldLocation.node(), fea.FieldLocation.point(), ...
    fea.RecoveryPolicy.shapeFunctionInterpolation(), ...
    fea.ObservationComponent.all(), fea.CoordinateFrame.globalFrame(), ...
    fea.SignConvention.native(), fea.ComplexRepresentation.real(), ...
    fea.SamplingPolicy.finalSample(), fea.ReductionPolicy.none(), ...
    fea.ObservationStorage.new( ...
        fea.SignalPrecision.real64(), ...
        fea.CompressionPolicy.lossless(), ...
        fea.RetentionPolicy.canonical(), ...
        fea.PartialAccessPolicy.manifestAndChunks()), ...
    fea.ObservationPurpose.visualization(), ...
    fea.ObservationCost.new(100000, 1048576, 1048576, 1048576, 0));
```

Sampling can select the final sample, every accepted increment, explicit or interval-based time,
frequency, or load-factor samples, mode indices, typed solver, derived-series, or accuracy events,
or a bounded adaptive policy. Event attribution is separate from the physical time, frequency,
load-factor, mode, or study-parameter coordinate: a solver event names its completed result or
failed attempt, a derived-series event names its source history, and an accuracy event names its
source report. A solver-owned nonlinear `firstYield` or `contactStatusChange` sample is detected
from the accepted constitutive or contact state and names that completed solver result. A
`convergenceFailure` sample can only name a failed attempt; rejected increments inside an otherwise
completed solve are not relabeled as failed solves. If the requested event never occurs, RunMat
publishes a valid history with zero samples. That distinction lets callers tell “observed no event”
from “the observation did not run.” For example, use
`fea.SamplingPolicy.event(fea.ObservationEvent.firstYield())` as the sampling argument of a
nonlinear displacement history. When observation evaluation is partitioned, each event occurrence
is a separate fan-out. The occurrence attribution is part of every task, contribution, and reduction
evidence identity, so workers cannot combine samples from different occurrences or substitute a
different completed result. An event that does not occur produces no partition tasks and retains the
same canonical empty-history meaning.
`fea.ReductionPolicy.none()` preserves the requested data. `fea.ReductionPolicy.new(...)` combines
a typed spatial reduction, series reduction, and extrema-tie policy. Spatial minimum and maximum
results retain the canonical node, element, or interpolated path position where the value occurs.
`firstCanonical` retains the first exact tie in canonical target order; `allLocations` retains every
exact tie. This rule is independent of worker count and partition completion order.

Finite-element integrals, root-mean-square values, and L2 norms use the admitted element or face
quadrature and publish a global result. `resultant` integrates vector fields directly, converts
boundary stress to Cauchy traction, and converts boundary pressure through its required explicit
outward- or inward-positive sign convention. A moment always names its reference point; there is no
implicit model origin:

```matlab
origin = fea.Position.meters([0 0 0]);
forceReduction = fea.ReductionPolicy.new( ...
    fea.SpatialReduction.resultant(), fea.SeriesReduction.none(), ...
    fea.ExtremaTiePolicy.firstCanonical());
momentReduction = fea.ReductionPolicy.new( ...
    fea.SpatialReduction.moment(origin), fea.SeriesReduction.none(), ...
    fea.ExtremaTiePolicy.firstCanonical());
```

The source quantity keeps its physical shape while the reduced result has its own derived shape:
stress and pressure resultants and moments are three-component vectors, and an L2 norm is scalar.
Partition tasks integrate only their owned cells or faces; canonical compensated reduction makes
the fan-in order deterministic for the admitted partition plan, with serial/partitioned numerical
equivalence governed by the observation tolerance. Recovered reaction forces are already discrete
nodal forces, so their resultants are sums rather than a second finite-element integral; moments sum
each nodal force crossed with its lever arm. Whole-model and exact-node reaction targets are
supported. Constraint-target reaction totals remain fail-closed until evaluation supplies the exact
executed constraint plan, which is the sole authority for constraint membership.

Series reductions operate on a completed canonical history, after any requested spatial reduction.
Minimum, maximum, absolute maximum, and envelope retain the exact source-history sample, physical
coordinate, and node, element, or interpolated point for every output lane. An envelope is the
typed pair of lower and upper fields; it is not a plotting instruction. Exact ties follow the same
`firstCanonical` or `allLocations` policy as spatial extrema.

Mean, root-mean-square, and population variance require an explicit weighting policy:

- `fea.SeriesWeighting.uniformSamples()` gives every retained sample equal weight. This is useful
  for modes and genuinely uniform sample grids.
- `fea.SeriesWeighting.trapezoidalIndependentDomain()` integrates piecewise linearly over physical
  time, frequency, load factor, or a named study parameter. It preserves nonuniform sample spacing
  and is rejected for static and discrete mode domains.

The statistical operation owns its weighting, so an unqualified `mean()` or
`rootMeanSquare()` cannot silently interpret a nonuniform history as uniform. For example:

```matlab
time_mean = fea.SeriesReduction.mean( ...
    fea.SeriesWeighting.trapezoidalIndependentDomain());

tip_history = fea.Observation.history( ...
    "tip_displacement_mean", ...
    fea.ObservationQuantity.displacement(), ...
    fea.ObservationTarget.point([0.10, 0.00, 0.00]), ...
    fea.FieldLocation.node(), fea.FieldLocation.point(), ...
    fea.RecoveryPolicy.shapeFunctionInterpolation(), ...
    fea.ObservationComponent.y(), fea.CoordinateFrame.globalFrame(), ...
    fea.SignConvention.native(), fea.ComplexRepresentation.real(), ...
    fea.SamplingPolicy.timeSeconds([0.0, 0.1, 0.4, 1.0]), ...
    fea.ReductionPolicy.new( ...
        fea.SpatialReduction.none(), time_mean, ...
        fea.ExtremaTiePolicy.firstCanonical()), ...
    fea.ObservationStorage.new( ...
        fea.SignalPrecision.real64(), ...
        fea.CompressionPolicy.lossless(), ...
        fea.RetentionPolicy.canonical(), ...
        fea.PartialAccessPolicy.manifestAndChunks()), ...
    fea.ObservationPurpose.publication(), ...
    fea.ObservationCost.new(100000, 1048576, 1048576, 1048576, 0));
```

A series reduction requires every produced sample. A history retained with `latestSamples(...)`
cannot be reduced after older samples have been discarded; RunMat rejects it rather than reporting
a statistic over an accidental suffix. Event histories cannot request a series reduction because
their inventory is allowed to be empty; callers can explicitly handle the event inventory and then
reduce a separately defined, nonempty history. Complex means are evaluated component-wise on the
complex values. Other complex statistics and extrema require an explicit real, imaginary,
magnitude, or phase view so their ordering and output domain are unambiguous.

Storage declares precision, compression, retention, and partial-access behavior. Retention may be
canonical, bounded to the latest or an explicit maximum number of samples, or last until dependent
steps complete. `manifestAndChunks` lets a local or remote consumer fetch metadata and only the
result chunks it needs. A retained history uses one semantic manifest payload and one independently
addressable payload per retained sample. The manifest resolves physical coordinates to exact sample
identities; local and remote importers then verify the root plus only that sample's chunks against a
sample-sized object and byte bound. Unrelated samples need not be present in the consuming cache and
are never gathered as a side effect. The declared `fea.ObservationCost` feeds resource admission;
it is not a measured result.

## Diagnostics and Monitors

Diagnostics use `fea.Observation.diagnostic(id, kind, sampling, retention)`. Typed kinds cover
convergence, balance, constraints, contact, adaptation, resource use, and execution equivalence.

A monitor reads a scalar observation already declared on the step:

```matlab
monitor = fea.Observation.monitor( ...
    "stop_on_residual", "residual_norm", ...
    fea.MonitorCadence.acceptedIncrements(5), ...
    fea.MonitorStopCondition.absoluteChangeBelow( ...
        fea.ObservationQuantity.residual(), 1e-8, 3));
```

Cadence is a positive accepted-increment interval or a wall-time interval in seconds. Stop
conditions are none, less-than, greater-than, or an absolute-change threshold sustained for a
declared number of samples. A monitor can stop work; it cannot modify a load, material, constraint,
or solver state.

## Accuracy Plans

An accuracy criterion has one target, error contribution, tolerance, required evidence, and
mandatory flag. Targets may name an observation, a quantity, global equilibrium, interface
closure, or execution equivalence. Relative tolerances are dimensionless; absolute tolerances pair
a typed quantity with a canonical-unit value.

Error contributions remain separate: geometry admission, boundary approximation, spatial,
algebraic, nonlinear, temporal, frequency-sampling, modal-truncation, stochastic, coupling,
interface-projection, field-transfer, signal-interpolation, recovery, observation-sampling,
precision, and execution-equivalence error are not added into a decorative total.

Required evidence may be an a-posteriori estimator, convergence study, conservation balance,
residual bound, reference comparison, statistical confidence, or transfer bound. When required
evidence is unavailable, the plan either fails or marks the result indeterminate and experimental.
When a resource ceiling is reached, the typed action may fail, refine, tighten the solver, extend
sampling, or warn only for explicitly experimental work.

`fea.AccuracyPlan.experimentalBaseline(id)` is intentionally non-production. It is useful while
building or diagnosing a formulation, but cannot make an experimental formulation publishable.
Numerical acceptance, formulation maturity, and publication status remain independent decisions.

## Selection Is Not Rendering

Observation requests decide what the solve computes and retains. After a run, `fea.field` selects
one published field. Ordinary plotting functions render the resulting object:

```matlab
results = fea.results(runResult);
stress = fea.field(results, "structural.von_mises");
plot(stress);
mesh(results);
```

`plot` does not select stress, choose a recovery method, or trigger an unplanned full-field
transfer. `mesh` displays geometry or solver topology and does not assign materials or define an
analysis region.

## Local, GPU, and Cluster Execution

Sequence, observation, and accuracy objects contain no host paths, device handles, worker ids, or
live callbacks. Planning converts the validated semantic sequence into execution tasks and selects
CPU, GPU, remote, or distributed placement under the normal RunMat execution contracts. Independent
frequency points, samples, branches, and partition-local observation work may scale out when the
selected formulation supports it. Global reductions retain collective and partition evidence.

The RunMat authoring boundary keeps those responsibilities explicit and typed:

```matlab
backend = fea.ComputeBackend.cpu();
requirement = fea.FormulationRequirement.experimental();
execution = fea.ExecutionPlan.defaults(sequence, backend, requirement);
study = fea.study("bracket", geometry, ...
    "Template", "linear_static_structural", ...
    "Model", model, ...
    "Execution", execution);
```

For a heterogeneous sequence, construct one `fea.SolverOptions` and `fea.ExecutionStep` per
procedure step, then pass the ordered entries to `fea.ExecutionPlan.new(sequence, steps)`. The
sequence remains the authority for physical order and state flow; the execution plan only binds
those steps to solver options, requested backend, and required formulation maturity. Cluster
selection belongs to the shared CLI/execution placement layer rather than either object.

Changing placement cannot weaken an accuracy criterion, alter activation, invent a state
transition, or turn an experimental formulation into a production one. Checkpoint and result
artifacts retain sequence, topology, partition, formulation, observation, and accuracy identity so
retries and remote execution remain attributable.

A sweep has one canonical definition containing its contract digest, ordered study identities, and
typed composition policy (`fail fast` or `collect all`). The native root derives the sweep identity
from that definition and rejects a reordered, omitted, duplicated, or substituted prepared study
before work begins. Worker completion order and retry placement therefore cannot change aggregate
identity or authored result order.

Remote sweep output is a manifest-first catalog. Importing the catalog validates the aggregate root,
ordered terminal statuses, and complete reference inventory without reading successful child-study
closures. A consumer may then materialize one study by authored index; only that study's mesh,
fields, observations, histories, and evidence are fetched. Full sweep materialization is an explicit
convenience operation over the same selective interface, not a requirement imposed by the driver.
