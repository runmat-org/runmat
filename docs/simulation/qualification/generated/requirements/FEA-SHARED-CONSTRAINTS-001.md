# FEA-SHARED-CONSTRAINTS-001: Typed constraint compilation, enforcement and reactions

0 PASS, 6 PARTIAL, 0 FAIL, 162 OPEN, 0 TAGGED. Generated; do not edit.

## constraint_system_scope

Evidence kind: SemanticRealization. Criterion: Constraints are a first-class system capability, not a special case inside boundary assembly. They affect topology resolution, algebra, nonlinear state, modal admissibility, dynamics, reactions, adaptation, partitioning, and result evidence.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## distinct_mathematical_roles

Evidence kind: SemanticRealization. Criterion: The public and internal models must distinguish: The existing cross-physics `BoundaryConditionKind` must be decomposed. A heat flux is not a kinematic constraint; acoustic impedance is not fixed displacement; contact is not a constrained boolean row.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## closed_domain_choices

Evidence kind: SemanticRealization. Criterion: Strings are permitted for authored identifiers and external lookup data: constraint IDs; region names and tags used to obtain a typed geometry selection; and user-facing labels. Strings are not permitted to choose: constraint kind; physics family; degree of freedom; coordinate frame kind; component selection; enforcement method; pairing method; schedule kind; or value dimension. The RunMat object ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## concrete_typed_object_roles

Evidence kind: SemanticRealization. Criterion: The final object surface should contain these typed roles, with exact class layout adjusted to the capabilities of RunMat's object system: Typed contract structure: fea.Constraint fea.StructuralConstraint fea.ThermalConstraint fea.ElectricalConstraint fea.MagneticConstraint fea.AcousticConstraint fea.FluidConstraint fea.Boundary fea.StructuralBoundary fea.ThermalBoundary fea.MagneticBoundary fea.A...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## compatible_supporting_types

Evidence kind: SemanticRealization. Criterion: A common interface permits heterogeneous storage and orchestration, but construction remains concrete and family-specific. Invalid combinations should be unrepresentable where practical and otherwise rejected at the earliest typed validation boundary.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## typed_authored_lowering

Evidence kind: SemanticRealization. Criterion: `runmat-analysis-core` should use nested typed enums and structures rather than a single large boundary enum or string tag dispatch: Typed contract structure: pub enum AnalysisConstraint { Structural(StructuralConstraint), Thermal(ThermalConstraint), Electrical(ElectricalConstraint), Magnetic(MagneticConstraint), Acoustic(AcousticConstraint), Fluid(FluidConstraint), } pub enum StructuralConstraint...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## exact_topology_targets

Evidence kind: SemanticRealization. Criterion: Resolved constraints must not retain an unqualified `region_id: String`. A typed target carries: admitted geometry identity and revision; persistent region or entity identity; entity dimension; selection digest; expected topology kind; selection/query provenance; and any cardinality or pairing requirements. The type and validation rules distinguish point, edge, face, volume, node, element, cell, a...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## governed_coordinate_frames

Evidence kind: SemanticRealization. Criterion: Constraint values may be expressed in: the global frame; a typed user Cartesian, cylindrical, or spherical frame; a material or element frame where the formulation supports it; or a boundary-normal/tangent frame. Frame transforms are resolved into canonical evidence. Boundary-normal constraints must reject ambiguous edge/corner selections or use an explicitly documented averaging/pairing policy. C...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## serializable_values_and_schedules

Evidence kind: SemanticRealization. Criterion: Constraint values may be: constant; step-dependent; a tabulated time history; frequency-dependent, including complex values where the family permits them; parameter-dependent through a portable study parameter; or supplied by a typed field artifact with compatible topology and units. Canonical schedules are serializable data. A live host closure does not enter the analysis model. RunMat callables ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## structural_capability_inventory

Evidence kind: SemanticRealization. Criterion: The structural production surface includes: fixed support; component-wise prescribed translation; component-wise prescribed rotation where rotational DOFs exist; normal displacement and roller support; symmetry and antisymmetry; rigid coupling to a reference point; distributing coupling to a reference point; ties over matching and supported nonmatching surfaces; general linear multipoint equations...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## cross_family_capability_inventory

Evidence kind: SemanticRealization. Criterion: Incompressibility remains part of the mixed fluid formulation. It must not be exposed as a generic user-authored constraint equation.

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## canonical_plan_and_affine_reduction

Evidence kind: SemanticRealization. Criterion: Authored equality constraints compile into an immutable canonical plan representing: Typed contract structure: C q = d and, where reduction is appropriate: Typed contract structure: q = T q_reduced + q_offset The plan contains: canonical constraint and row identities; typed participating DOFs; sparse equation rows; prescribed values or schedule references; target provenance; coordinate transformat...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::constraints::compilation::admission::tests::exact_caller_boundary_preserves_original_plan_across_all_compiler_routes` (crates/runmat-analysis/fea/src/constraints/compilation/admission/tests.rs:10) - Prepared structural constraint fixtures compare original canonical plans and bytes at the exact requested workspace boundary and reject one byte below; no cross...<br>`runmat-analysis-fea[lib]::constraints::tests::general_equation_compiles_to_one_canonical_sparse_row` (crates/runmat-analysis/fea/src/constraints/tests.rs:674) - A general equation compiles to one canonical sparse row.<br>`runmat-analysis-fea[lib]::constraints::tests::canonical_solid_solve_enforces_affine_equation_and_recovers_reactions` (crates/runmat-analysis/fea/src/constraints/tests.rs:824) - A canonical solid solve enforces an affine equation and recovers reactions. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## compatible_enforcement

Evidence kind: SemanticRealization. Criterion: Use the mathematical structure of the relation to select enforcement: direct elimination or affine lifting for ordinary prescribed values; transformation or null-space reduction for structured ties, periodic relations, rigid couplings, and compatible multipoint equations; Lagrange multipliers for exact general relations where reduction is unsuitable; formulation-specific null-space or gauge handli...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::constraints::tests::dense_global_equation_selects_multiplier_enforcement_before_affine_fill` (crates/runmat-analysis/fea/src/constraints/tests.rs:723) - A dense global equation selects multiplier enforcement before any affine fill; direct elimination and penalty selection are separate cases. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## nonlinear_inequality_state

Evidence kind: SemanticRealization. Criterion: Contact remains an `AnalysisInterface` at the authored physical-model boundary and compiles into a nonlinear inequality constraint system. It does not compile into the equality matrix `C`. The contact system owns: surface pairing and search updates; gap and normal definitions; active-set or complementarity state; normal enforcement; tangential/friction state; residual and tangent contributions; hi...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::constraints::tests::nonlinear_accepted_states_preserve_complete_constraint_kinematics` (crates/runmat-analysis/fea/src/constraints/tests.rs:1510) - Nonlinear accepted states preserve complete constraint kinematics. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## attributable_validation

Evidence kind: SemanticRealization. Criterion: Constraint validation has two levels. Before assembly, structural validation detects: empty or unmatched targets; invalid topological dimensions; constraints on unavailable DOFs; ambiguous frames or normals; conflicting prescribed values; obviously duplicate or cyclic relations; invalid or non-bijective periodic pairing where bijection is required; inconsistent transforms around relation cycles; i...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::physics::structural::topology::identity::tests::counted_validation_reports_first_authored_failure_before_later_duplicates` (crates/runmat-analysis/fea/src/physics/structural/topology/identity/tests.rs:193) - Counted validation reports the first authored failure before later duplicates. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## original_reaction_and_work

Evidence kind: SemanticRealization. Criterion: Constraint results include: nodal reaction force and moment; face, region, and reference-point resultants; multiplier values where applicable; generalized reaction per authored constraint; constraint work and energy; relation violation norms; prescribed-value error; action-reaction balance for ties and couplings; active, redundant, and eliminated row counts; and rank/null-space diagnostics. Reacti...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::solve::reaction::tests::independent_constraint_rows_match_the_diagonal_reaction_law_across_workers` (crates/runmat-analysis/fea/src/solve/reaction.rs:231) - Independent constraint rows match the diagonal reaction law across workers.<br>`runmat-analysis-fea[lib]::solve::static_balance::tests::prescribed_beam_extension_closes_reaction_and_work_against_closed_form` (crates/runmat-analysis/fea/src/solve/static_balance.rs:445) - A prescribed beam extension closes reaction and work against the closed form. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## recovery_after_adaptation

Evidence kind: SemanticRealization. Criterion: Constraints bind to persistent geometry and topology identities rather than incidental node numbers. After h- or p-adaptation, the analysis projection rebuilds the canonical constraint plan, checks target coverage and pairing, transfers necessary state, and records old/new plan lineage. For large-deformation and contact studies, geometry-dependent normals, pairings, and active state update through...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::adaptation::iteration::tests::constraint_lineage::persistent_clamp_rebuilds_new_midside_constraints_without_numeric_node_aliases` (crates/runmat-analysis/fea/src/adaptation/iteration/tests/constraint_lineage.rs:24) - A persistent clamp rebuilds new midside constraints without numeric node aliases after refinement. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## partition_owned_rows

Evidence kind: SemanticRealization. Criterion: Every canonical constraint row has one stable identity and one publication owner. Participating DOFs may span partitions; required ghost/halo data is explicit. Distributed assembly rejects missing, duplicate, or conflicting rows. Rigid couplings, global equations, and reference points declare collective communication requirements. Partition-local elimination cannot change the mathematical constrai...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## governed_constraint_corpus

Evidence kind: SemanticRealization. Criterion: Required governed cases include: component-wise zero and nonzero prescribed values; constraints in rotated and boundary-normal frames; rigid-body mode removal without spurious stiffness; intentionally underconstrained and overconstrained systems; duplicate and linearly dependent equations; rigid and distributing reference-point couplings with force/moment balance; matching and nonmatching ties; pe...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## constraint_mathematical_role_contract

Evidence kind: SemanticRealization. Criterion: Essential field constraint; Prescribed displacement, rotation, temperature, electric potential, pressure reference; Restricts one or more solution DOFs.. Natural or Robin boundary; Traction, heat flux, convection, acoustic impedance, radiation; Adds residual, operator, or source terms on a boundary.. Kinematic relation; Tie, periodicity, rigid coupling, symmetry, multipoint equation; Relates multi...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## family_constraint_contract

Evidence kind: SemanticRealization. Criterion: Thermal; Prescribed temperature, tied temperature, periodic temperature, and supported affine temperature jumps.. Electrostatic/conduction; Prescribed potential, equipotential terminals, floating conductors, integral charge/current conditions, and gauge/reference handling.. Magnetostatic/Maxwell; Gauge conditions, symmetry, periodicity, and complex phase-shifted relations where supported.. Acousti...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

