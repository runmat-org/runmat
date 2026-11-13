# FEA-SHARED-EXCITATION-001: Typed signals, frozen generators and conservative actions

0 PASS, 5 PARTIAL, 0 FAIL, 99 OPEN, 0 TAGGED. Generated; do not edit.

## one_cross_family_authority

Evidence kind: SemanticRealization. Criterion: Excitation is a first-class cross-family system. It must not be implemented independently by structural dynamics, thermal, electromagnetic, acoustic, and fluid modules.

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

## distinct_layer_and_action_ownership

Evidence kind: SemanticRealization. Criterion: Four layers have distinct ownership: The solver never depends on `runmat-value`, the VM, a session workspace, or callback dispatch. Runtime adapters may accept a callable `Value`; the validated analysis input is data. The authored model keeps these concepts separate: an essential constraint may reference a signal, but remains a constraint; a natural boundary or volume action is a load or source ex...

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

## dimensioned_signal_contract

Evidence kind: SemanticRealization. Criterion: A signal declares: stable signal identity; independent domain: constant, time, frequency, or portable study parameter; physical quantity and dimensions; scalar, vector, tensor, complex, or topology-field value shape; coordinate frame and component meaning when applicable; representation: constant, tabulated samples, a supported standard waveform, or an immutable artifact; abscissa and ordinate uni...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-core[lib]::validate_signals::topology_field_tests::field_validation_preserves_frames_precision_and_approximation_evidence` (crates/runmat-analysis/core/src/validate_signals/topology_field_tests.rs:222) - Field validation preserves frames, precision and approximation evidence.<br>`runmat-analysis-fea[lib]::excitation::tests::step_interpolation_selects_the_documented_side_of_an_interval` (crates/runmat-analysis/fea/src/excitation/tests.rs:323) - Typed tabulated force signal at time 0.5 selects distinct left and right endpoint vectors; no generator freezing, spatial integration or placement equivalence. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## conservative_spatial_realization

Evidence kind: SemanticRealization. Criterion: An excitation binds a signal to physical space and declares: stable excitation identity and containing excitation-case identity; typed action such as force, moment, traction, pressure, body force, base displacement/velocity/acceleration, heat generation, heat flux, ambient temperature, current density, terminal voltage, acoustic pressure, normal velocity, inlet velocity, or mass flow; persistent r...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::excitation::tests::realization_preserves_exact_multi_entity_region_targets` (crates/runmat-analysis/fea/src/excitation/tests.rs:144) - Realization preserves exact multi-entity region targets.<br>`runmat-analysis-fea[lib]::excitation::tests::preserves_reference_point_force_and_moment_action_identity` (crates/runmat-analysis/fea/src/excitation/tests.rs:645) - Reference-point force and moment action identity is preserved. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## frozen_bounded_generators

Evidence kind: SemanticRealization. Criterion: RunMat code may construct a signal from ordinary arrays or from a function handle or anonymous closure. Array construction is the simplest path and naturally composes with RunMat signal processing, control systems, data access, sparse values, parallel execution, and GPU placement. A callable-backed constructor additionally requires a typed domain and materialization policy. The preferred callable ...

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

## effect_policy_and_runtime_values

Evidence kind: SemanticRealization. Criterion: The initial production effect policy rejects workspace writes, environment mutation, undeclared filesystem access, network access, UI interaction, uncontrolled randomness, clock dependence, host callbacks, and unknown effects. Immutable portable captures are allowed. Random excitation uses an explicit seed, distribution, PSD/cross-PSD, or frozen realization. Session-bound handles, mutable handle o...

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

## named_function_ports

Evidence kind: SemanticRealization. Criterion: A data-only `.fea` remains fully standalone. It may contain bounded inline samples or reference a declared data object that is content-hashed and frozen into the study bundle. The declarative form resolves once into the same typed objects. It imports the generating function from its actual `.m` file rather than relying on a magic namespace: Base acceleration is prescribed support kinematics rather...

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

## explicit_inputs_and_spectra

Evidence kind: SemanticRealization. Criterion: Initial conditions are model-owned reusable physical state and are selected by the consuming analysis; a reusable model never hard-codes a study step name. The closed kinds cover structural displacement, velocity, and angular velocity; thermal temperature; electric potential; magnetic vector potential; acoustic pressure; and fluid pressure and velocity. A condition has exactly one strongly typed s...

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

## distinct_modal_response_procedures

Evidence kind: SemanticRealization. Criterion: Modal eigenvalue extraction has no excitation. Forced use of modes is represented by separate procedures: modal transient response from time histories; modal harmonic response from complex frequency-domain excitations; modal response-spectrum analysis from a typed design or shock response spectrum; and random vibration response from PSD and cross-PSD inputs. Computing a shock response spectrum fro...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-core[lib]::validate_sequence::tests::modal_response_requires_typed_existing_spectrum_ids` (crates/runmat-analysis/core/src/validate_sequence/tests.rs:124) - Modal response requires typed existing spectrum ids.<br>`runmat-analysis-fea[lib]::solve::dynamics::response_spectrum::tests::modal_and_directional_combination_policies_have_independent_semantics` (crates/runmat-analysis/fea/src/solve/dynamics/response_spectrum/tests.rs:24) - Modal and directional combination policies have independent semantics. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## layered_generator_realization_identity

Evidence kind: SemanticRealization. Criterion: Generated physical input has layered identity: Typed contract structure: generator provenance = program revision + callable identity + captures + declared frozen inputs signal realization = canonical samples/field + quantity + units + interpolation + error evidence solver excitation = signal realization + spatial projection + activation + procedure Study provenance includes the generator identity ...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::execution::tests::generator_capture_and_realized_content_invalidate_the_realization` (crates/runmat-analysis/fea/src/execution/tests.rs:83) - Generator capture and realized content invalidate the realization.<br>`runmat-analysis-fea[lib]::execution::tests::canonical_signal_realization_round_trips_and_layers_content_identity` (crates/runmat-analysis/fea/src/execution/tests.rs:669) - Canonical signal realization round-trips and layers content identity. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## chunked_reuse_and_compiled_law_boundary

Evidence kind: SemanticRealization. Criterion: State-dependent executable laws require a separate future contract with a portable compiled ABI, typed state and outputs, derivative/tangent requirements, effect and capability policy, CPU/GPU/distributed implementations, checkpoint semantics, and verification. No family may introduce such callbacks ad hoc through the excitation API.

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

## governed_excitation_corpus

Evidence kind: SemanticRealization. Criterion: Required governed cases include: constant, piecewise-linear, step, and complex frequency-dependent signals; discontinuities, domain boundaries, interpolation, extrapolation rejection, and adaptive materialization error; closure capture and named project function equivalence to an explicit table; source/capture changes invalidating identity and identical canonical realizations reusing numerical wor...

| Formulation | Mesh | Execution | Status | Tests (boundary) |
|---|---|---|---|---|
| shared_infrastructure | canonical_mixed_topology | numerical_kernel | PARTIAL | `runmat-analysis-fea[lib]::excitation::tests::static_excitation_accepts_only_coordinate_independent_tables` (crates/runmat-analysis/fea/src/excitation/tests.rs:283) - Static excitation accepts only coordinate-independent tables; the constant, step and frequency-dependent cases of the corpus are not all covered. |
| shared_infrastructure | canonical_mixed_topology | public_study | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | native_process | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | durable | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | remote_execution | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | parallel_cpu | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | domain_decomposed | OPEN |  |
| shared_infrastructure | canonical_mixed_topology | claimed_accelerator | OPEN |  |

## input_layer_ownership_contract

Evidence kind: SemanticRealization. Criterion: RunMat language and general numerical libraries; Generate, import, filter, resample, transform, and combine arrays using ordinary functions, closures, signal-processing operations, control-system models, data APIs, and accelerator execution.. `runmat-analysis-core`; Own strongly typed signals, physical quantities, excitations, initial conditions, response spectra, targets, frames, activation, and ...

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

