# Reviewed numerical boundaries

These entries describe narrow internal algorithms. They do not exempt their physical producers, public contracts, scalar policy fields, results, or other files in the same crate.

## Generic numerical kernels

- `fea/src/algebra/reference.rs`: deterministic sparse matrix/vector arithmetic (`dot_canonical`, scoped dot/AXPY/SpMV, residual). It performs no physical unit interpretation. Full structural calls retain `AssemblySummary.structural_dof_layout`, `StructuralPhysicalOperator`, and `StructuralSolveLayout`; modal calls retain `ModalBasisCoordinates` and the work-conjugate projection. Thermal/electrical/etc. physical owner admission is inventoried separately; approving generic arithmetic does not declare those migrations complete.
- `fea/src/algebra/anchored_energy/outward.rs`: outward-rounded interval arithmetic. Interval endpoints are mathematical operands, not an independent energy unit authority.
- `meshing/core/src/quality/predicate/vector.rs`: abstract vector dot/cross/norm. Positions entering meshing owner contracts are canonical typed `Quantity<Position, Storage>`; direction and cross-product results are deliberately not mislabeled Position.
- `meshing/core/src/quality/predicate/exact_dyadic.rs`: robust predicate significand/exponent arithmetic. Exact bit representations and predicate error bounds are mathematical representation data.

## Canonical bit identity

`fea/src/physics/structural/topology/identity.rs::NodeContract.coordinate_bits` serializes checked `StructuralNode.coordinates_m` as IEEE `u64` bit patterns under the established `coordinates_m` wire key. This is an identity encoding, not a physical position storage owner. A coordinate codemod initially wrapped this legacy spelling; review corrected it and excluded integer bit arrays from subsequent mappings.

## Recovered field storage

The authoritative `RecoveredFieldDefinition` retains quantity, shape, association, complex lane representation and recovery derivation. `physical_layout` projects that existing header into the shared `QuantityLayout`; central storage admission checks finite host values through `QuantityField`. Real and imaginary lanes carry the same physical descriptor. Device buffers retain the checked header and cardinality; device-value validity must be established by their executing backend.

A bare generic Eigenvalue/Residual/Balance/ConstraintViolation name cannot choose a unit. Active field recovery producers were audited in `results`/`pipeline` and emit concrete quantities. Modal and buckling scalar result contracts are separate. Future generic field producers must provide a physics/procedure-specific descriptor; they cannot obtain an invented dimensionless header from a diagnostic name.

## Modal convention

The basis carries full translational/rotational DOF layout and a canonical reference generalized mass of 1 kg. Generalized coordinates are Length/Displacement; translation shape entries are ratios and rotation shape entries are rad/m. Reduced K/M/C have stiffness/mass/damping kinds. `project_load` consumes a checked Force/Moment field and returns modal Force; state projection/reconstruction preserve derivative roles. Durable generalized displacement, velocity and acceleration use the same shared Quantity storage. Dense Newmark scalar arithmetic is an explicit internal extraction, not a second unit representation. Resource admission counts the full retained layout once, not once per mode.

Compilation and cross-boundary qualification remain pending for the latest source cutover. This document is not completion evidence.

## Scalar physics and numerical preparation

`PhysicalScalarSystem<K>` now owns the numerical system, state kind and actual free-equation descriptor. Its `rhs_field` retains prescribed rows in state units and free rows in forcing units. Physics admission compares the retained equation descriptor against its actual owner before numerical execution. Original physical operators have homogeneous forcing units; identity-row substitution is a separate, explicitly mixed numerical preparation. The generic sparse kernel is reached only through the owned physical system's scoped canonical view, and its result is admitted into `LinearSolveResult<Quantity<K, Vec>>` with cardinality/finite checks.

Thermal state is TemperatureDifference while solving/integrating departures. Steady reconstruction consumes that buffer and explicitly creates Temperature storage; diagnostic-only retained results release the state and retain `LinearSolveResult<()>` metrics. Thermal capacity belongs to a first-derivative energy-storage equation, not structural second-order mass.

Electrical material coefficients retain the equation's actual ElectricPermittivity or ElectricalConductivity kind. Dividing the free equation by that coefficient leaves Potential×Length rows; prescribed rows remain Potential. The previous dimensionless-matrix claim was corrected. Response/receipt flux and work descriptors are checked against the electrical equation (Charge/Current and Energy/Power). Typed vacuum constants preserve the prior solver values under one shared representation.

Frequency→AngularFrequency uses an explicit 2π conversion and retains nominal radians. Unit admission rejects Hz for a rad/s field despite matching SI exponents.

### Generic algebra representation and coefficients

The CSR container (`algebra/matrix.rs`) and triplet assembler (`algebra/matrix/assembler.rs`) are numerical storage, with no claim that a matrix has one physical unit. `QuantityLinearOperator`, `StructuralPhysicalOperator`, `StructuralSolveLayout` and `PhysicalScalarSystem` retain the actual input/output component layouts around this storage. Coefficients describe output/input, including nominal angular powers; mixed Dirichlet, structural and gauge rows cannot be assigned a single matrix dimension. This disposition covers the generic container and algorithm declarations only, and does not approve an unfinished physics owner's admission seam.

`CompensatedSum` is generic Neumaier accumulation. `right_inverse.rs` certifies a mathematical CSR inverse via an identity-product bound. `normalized_vector.rs` computes orientations and directional differentials for caller-supplied perturbation parameters. These kernels introduce no physical unit conversion or independent dimensional floor. Raw parameters here are numerical polymorphism, not a default physical QuantityKind.

## Magnetostatic gauge and harmonic weak equations

The magnetic saddle retains `MagneticUnknownLayout`: free edge circulations are Wb; gauge coordinates retain the explicit conversion from A/m through reference permeability and squared reference length into Wb. The scaled saddle maps these coordinates into Wb/m, while the original physical curl-curl operator maps Wb to A. Both use `QuantityLinearOperator`; original load and scaled load buffers retain separate kinds. GMRES is a scoped numerical operation under that owner and its output is admitted into the same layout before physical/gauge residual and field recovery. This normalization is neither an arbitrary frequency nor a dimensionless matrix label.

Maxwell assembly retains Potential for both real/imaginary oriented edge lanes and Current/Time for weak forcing. The original physical block's coefficient descriptor is output/input. The complex exponential's phase derivative explicitly cancels nominal radians through `phasor_derivative_rate`, rather than relabeling rad/s as Hz. Acoustic pressure K/M/B map Pressure into Volume/Time², Volume, and Volume/Time respectively. Their combination uses the same explicit phase time generator. Acoustic source, prescribed pressure, medium arrays and retained phase-reduced pressure coordinates carry the shared Quantity/QuantityField representation. The ratio-valued phase basis and pressure lifting are retained once through Arc and checked for shape/finite values; warm starts from another basis are rejected.

These source changes still require aggregate diagnostics and final tests; they are not completion evidence.

### Krylov and canonical insertion-order frames

`solve/krylov.rs`, `right_ilu.rs`, `right_diagonal.rs`, and `fixed_point.rs` are generic mathematical recurrences/preconditioners/proposals. Their owned f64 buffers are not physical field contracts. The inspected right scaling/factorization rejects noninvertible/nonfinite pivots without dimensional floors or pivot substitution; GMRES uses the caller's original residual and relative tolerance. Physical callers retain checked operators/coordinate layouts and perform original-equation acceptance. An Anderson proposal's acceptance remains with its physical owner; the generic helper does not authorize mixing dimensional residual lanes.

`meshing/tetrahedron/cdt/volume_mesh/frame.rs` derives a private insertion-order frame only from `DelaunayConstraints.nodes` typed Position coordinates. Extracted origin/radius and axes form temporary normalized ordering hints; quantization affects insertion order, while predicates, metric evaluation and published nodes retain original physical coordinates. These raw frame internals are a scoped canonical geometric algorithm, not imported unit evidence or a persisted geometry contract.

## Experimental coupling proxy exclusion

The legacy `assembly/mod.rs` electro-thermal context branch computes a clamped, index-shaped synthetic RHS; `pipeline/electro_thermal.rs` reconstructs proxy fields from that scale rather than solving the physical electrical/thermal equations. It is not a valid numerical boundary at which to assign a physical quantity descriptor. Production structural admission now rejects this context explicitly. The remaining proxy is reachable only through the explicitly named `ExperimentalSyntheticOperator` formulation, and its coefficients/diagnostic recovery must not be advertised as physical execution. Prescribed solid thermomechanical loads and solved typed nodal thermal transfers remain supported through `thermoelastic_load::{assemble_prescribed_temperature_load,assemble_solved_temperature_load}`. Prescribed thermomechanical context without a supported physical solid mesh is rejected outside the explicit experimental formulation.
