use std::collections::BTreeSet;

use runmat_analysis_core::{AccuracyResourceCeiling, AnalysisModel};
use runmat_execution::{
    resource::{AcceleratorRequest, Capability, ResourceInventory, ResourceUsage},
    Digest,
};
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, PersistentEntityId,
    SolverMeshArtifact, StableDigest,
};

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    fixtures::{fixture_model, FixtureId},
    results::observation::{resolve_structural_observation_plan, ObservationLimits},
    topology::AnalysisProjection,
};

use super::*;

mod compiler_phase;
mod curved_connector;
mod exact_work;
mod excitation_integration;
mod general;
mod prepared_inputs;
mod thermal_state;

#[test]
fn static_catalog_construction_cannot_be_underreserved() {
    let mut planned = input(PartitioningPolicy::Monolithic);
    let FeaResourceDemand::StructuralLinearPipeline(demand) = &mut planned.demand else {
        panic!("static demand")
    };
    demand
        .state_profiles_output
        .construction_resident_bytes_admission_budget
        .0 -= 1;
    assert!(build_resource_plan(planned).is_err());
    let planned = input(PartitioningPolicy::Monolithic);
    assert!(planned
        .demand
        .output_facts(crate::execution::AnalysisNumericalObjectKind::StatePublicationProfiles)
        .is_some());
    assert!(
        !planned.demand.governs_prepared_outputs(),
        "legacy result/receipt admission remains separate debt"
    );
}

fn input(partitioning: PartitioningPolicy) -> FeaResourcePlanningInput {
    FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        phase: FeaResourcePhase::StructuralLinearPipeline,
        demand: FeaResourceDemand::StructuralLinearPipeline(
            StructuralLinearPipelineResourceDemand {
                constraint_compilation_peak_bytes: ByteCount(1),
                inputs: None,
                state_profiles_output: PreparedOutputResourceFacts {
                    encoded_payload_bytes_upper_bound: ByteCount(1),
                    transport_closure_bytes_upper_bound: ByteCount(1),
                    construction_resident_bytes_admission_budget: ByteCount(65_540),
                },
                operator: StructuralOperatorResourceDemand {
                    excitation_workspace: Default::default(),
                    topology_digest: Digest::sha256("topology"),
                    numerical_scope_digest: Digest::sha256("pattern"),
                    topology: TopologyResourceFacts {
                        nodes: Count(4),
                        elements: Count(1),
                        boundary_faces: Count(4),
                        element_node_references: Count(4),
                        boundary_node_references: Count(12),
                        node_neighbor_references: Count(12),
                        estimated_resident_bytes: ByteCount(1_088),
                    },
                    sparse: SparseResourceFacts {
                        rows: Count(4),
                        columns: Count(4),
                        operator_nonzeros: vec![Count(10), Count(4), Count(4)],
                    },
                    constraints: ConstraintResourceFacts {
                        rows: Count(1),
                        coefficients: Count(1),
                        reference_points: Count(0),
                        estimated_resident_bytes: ByteCount(120),
                    },
                    assembly: AssemblyResourcePolicy {
                        contribution_nonzero_upper_bound: Count(10),
                        contribution_bytes_upper_bound: ByteCount(240),
                        scratch_bytes_per_element: ByteCount(10),
                        fixed_scratch_bytes_per_worker: ByteCount(20),
                    },
                    solve: SolveResourcePolicy::Iterative {
                        vector_count: Count(5),
                        scalar: NumericalScalar::RealF64,
                        preconditioner: PreconditionerStorage::Diagonal,
                        transpose_index_bytes_upper_bound: ByteCount(0),
                        path: StaticSolvePath::ProvingFactorization,
                        admission: SolverAdmissionResourcePolicy {
                            dimension: Count(4),
                            factor_nonzero_upper_bound: Count(10),
                            update_count_upper_bound: Count(12),
                            resident_bytes_upper_bound: ByteCount(200),
                        },
                    },
                    phase_resources: PhaseResourcePolicy {
                        resident_value_count: Count(4),
                        resident_bytes_upper_bound: ByteCount(32),
                        scratch_bytes: ByteCount(40),
                        retained_bytes: ByteCount(32),
                    },
                    field_recovery: Default::default(),
                    observations: ObservationResourceFacts {
                        maximum_samples_per_observation: Count(1),
                        produced_sample_count: Count(1),
                        retained_sample_count: Count(1),
                        operation_count: Count(100),
                        peak_resident_bytes: ByteCount(64),
                        stored_bytes: ByteCount(200),
                        transfer_bytes: ByteCount(150),
                        collective_count: Count(0),
                    },
                    publication: CheckpointArtifactPolicy {
                        result_artifact_bytes: ByteCount(33),
                        checkpoint_bytes: ByteCount(50),
                        artifact_overhead_bytes: ByteCount(20),
                        relay_bytes: ByteCount(12),
                        result_egress_bytes: ByteCount(30),
                        checkpoint_egress_bytes: ByteCount(40),
                    },
                    partitioning,
                    accuracy_demand: AccuracyResourceDemand {
                        refinement_cycles: Count(0),
                        solver_tightenings: Count(0),
                        samples: Count(1),
                    },
                },
            },
        ),
        output_set: None,
        worker: WorkerResourcePolicy {
            cpu_millicores: 1_000,
            maximum_wall_milliseconds: 1_000,
            fixed_memory_bytes: ByteCount(100),
            fixed_scratch_bytes: ByteCount(5),
            required_capabilities: BTreeSet::from([Capability::NetworkDenied]),
            accelerators: Vec::new(),
        },
        accuracy_ceiling: AccuracyResourceCeiling {
            maximum_refinement_cycles: 1,
            maximum_solver_tightenings: 1,
            maximum_samples: 4,
            maximum_wall_time_milliseconds: 10_000,
            maximum_resident_bytes: Some(64 * 1024 * 1024),
        },
    }
}

fn operator_mut(input: &mut FeaResourcePlanningInput) -> &mut StructuralOperatorResourceDemand {
    input
        .structural_operator_mut()
        .expect("test input is an operator demand")
}

#[test]
fn affine_transpose_index_bytes_are_included_in_solve_memory_and_identity() {
    let mut baseline_input = input(PartitioningPolicy::Monolithic);
    // Exercise a solve-dominant peak; the small fixture now correctly peaks at catalog
    // construction, which does not overlap the transpose-index workspace.
    let SolveResourcePolicy::Iterative { vector_count, .. } =
        &mut operator_mut(&mut baseline_input).solve
    else {
        panic!("iterative fixture")
    };
    *vector_count = Count(10_000);
    let base = build_resource_plan(baseline_input.clone()).unwrap();
    let mut indexed_input = baseline_input;
    let SolveResourcePolicy::Iterative {
        transpose_index_bytes_upper_bound,
        ..
    } = &mut operator_mut(&mut indexed_input).solve
    else {
        panic!("fixture must use an iterative solve");
    };
    *transpose_index_bytes_upper_bound = ByteCount(1_000_000);
    let indexed = build_resource_plan(indexed_input).unwrap();
    assert_ne!(indexed.identity, base.identity);
    assert_eq!(
        indexed.request.memory_bytes,
        base.request.memory_bytes + 1_000_000
    );
}

#[test]
fn saddle_gmres_charges_workspace_and_admission_without_preconditioner() {
    let mut planned_input = input(PartitioningPolicy::Monolithic);
    let admission = match &operator_mut(&mut planned_input).solve {
        SolveResourcePolicy::Iterative { admission, .. } => admission.clone(),
        _ => unreachable!(),
    };
    operator_mut(&mut planned_input).solve = SolveResourcePolicy::SaddlePointGmres {
        block_dimension: Count(5),
        restart: Count(5),
        maximum_iterations: Count(256),
        workspace_bytes_upper_bound: ByteCount(4_096),
        admission,
    };
    let plan = build_resource_plan(planned_input.clone()).unwrap();
    assert_eq!(plan.input, planned_input);
    assert_eq!(plan.breakdown.workspace_bytes, ByteCount(4_096));
    assert_eq!(plan.breakdown.preconditioner_bytes, ByteCount(0));
    assert_eq!(plan.breakdown.factorization_bytes, ByteCount(200));

    let mut wrong_dimension = planned_input.clone();
    if let SolveResourcePolicy::SaddlePointGmres {
        block_dimension, ..
    } = &mut operator_mut(&mut wrong_dimension).solve
    {
        *block_dimension = Count(4);
    }
    assert!(build_resource_plan(wrong_dimension).is_err());

    let mut underpriced_workspace = planned_input.clone();
    if let SolveResourcePolicy::SaddlePointGmres {
        workspace_bytes_upper_bound,
        ..
    } = &mut operator_mut(&mut underpriced_workspace).solve
    {
        *workspace_bytes_upper_bound = ByteCount(8 * 5);
    }
    assert!(build_resource_plan(underpriced_workspace).is_err());

    let mut missing_proof = planned_input;
    if let SolveResourcePolicy::SaddlePointGmres { admission, .. } =
        &mut operator_mut(&mut missing_proof).solve
    {
        admission.resident_bytes_upper_bound = ByteCount(0);
    }
    assert!(build_resource_plan(missing_proof).is_err());
}

fn inventory(memory_bytes: u64, scratch_bytes: u64) -> ResourceInventory {
    ResourceInventory {
        cpu_millicores: 2_000,
        memory_bytes,
        scratch_bytes,
        accelerators: Vec::new(),
        capabilities: BTreeSet::from([Capability::NetworkDenied]),
    }
}

fn structural_fixture() -> (
    AnalysisModel,
    SolverMeshArtifact,
    AnalysisProjection,
    CanonicalStructuralConstraintPlan,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.push(PersistentEntityId::new(
            runmat_meshing_core::PersistentEntityKind::Face,
            "root".into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let constraints = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    (model, mesh, projection, constraints)
}

fn mixed_structural_fixture() -> (
    AnalysisModel,
    SolverMeshArtifact,
    AnalysisProjection,
    CanonicalStructuralConstraintPlan,
) {
    let mut mesh = crate::assembly::solver_solid::tests::artifact(ElementOrder::Tet4);
    let attached_coordinates = mesh.topology.nodes[0].coordinates_m.canonical_value();
    mesh.topology.nodes[0]
        .provenance
        .push(PersistentEntityId::new(
            runmat_meshing_core::PersistentEntityKind::Vertex,
            "beam_root".into(),
            vec!["root".into()],
        ));
    mesh.topology.nodes[0].provenance.sort();
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.push(PersistentEntityId::new(
            runmat_meshing_core::PersistentEntityKind::Face,
            "root".into(),
            vec!["root".into()],
        ));
        face.provenance.sort();
    }
    mesh.topology.boundary_faces[1]
        .provenance
        .push(PersistentEntityId::new(
            runmat_meshing_core::PersistentEntityKind::Face,
            "solid_load".into(),
            vec!["root".into()],
        ));
    mesh.topology.boundary_faces[1].provenance.sort();
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();

    let mut model = crate::fixtures::mixed_solid_beam_model(attached_coordinates);
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
    model.connectors = vec![runmat_analysis_core::AnalysisConnector {
        connector_id: "solid_to_beam_spring".into(),
        kind: runmat_analysis_core::ConnectorKind::AxialSpring {
            primary: runmat_analysis_core::RegionTarget::point("node:102"),
            secondary: runmat_analysis_core::RegionTarget::point(format!(
                "node:{}",
                mesh.topology.nodes[2].node_id
            )),
            stiffness: runmat_analysis_core::Stiffness::new(2_000.0),
        },
    }];
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let constraints = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    (model, mesh, projection, constraints)
}

#[test]
fn foundation_preflight_charges_selected_quadrature_and_retained_plan_copies() {
    let (mut model, mut mesh, _, _) = structural_fixture();
    let selected = PersistentEntityId::new(
        runmat_meshing_core::PersistentEntityKind::Face,
        "selected_foundation_face".into(),
        vec!["root".into()],
    );
    mesh.topology.boundary_faces[0].provenance.push(selected);
    mesh.topology.boundary_faces[0].provenance.sort();
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let foundation = runmat_analysis_core::AnalysisConnector {
        connector_id: "foundation_a".into(),
        kind: runmat_analysis_core::ConnectorKind::SurfaceNormalFoundation {
            target: runmat_analysis_core::RegionTarget::boundary(
                runmat_meshing_core::fixtures::fixture_face("selected_foundation_face"),
            ),
            modulus: runmat_analysis_core::FoundationModulus::new(1.0),
        },
    };
    model.connectors.push(foundation.clone());
    let one = connector_buffers::estimate_connector_buffers(
        &model,
        Some(&projection),
        projection.nodes.len() as u64,
    )
    .unwrap();
    let canonical = 7 * 1_024;
    let transient = 7 * 512 + 1_024;
    let metadata = 4_096 + 4 * "foundation_a".len() as u64;
    assert_eq!(one.assembly_peak_bytes, canonical + metadata + transient);
    assert_eq!(
        one.recovery_retained_bytes,
        2 * (canonical + metadata) + 1_024
    );
    assert_eq!(one.recovery_value_count, 16);

    let mut second = foundation;
    second.connector_id = "foundation_b".into();
    model.connectors.push(second);
    let two = connector_buffers::estimate_connector_buffers(
        &model,
        Some(&projection),
        projection.nodes.len() as u64,
    )
    .unwrap();
    assert_eq!(
        two.assembly_peak_bytes,
        2 * (canonical + metadata) + transient
    );
    assert_eq!(
        two.recovery_retained_bytes,
        4 * (canonical + metadata) + 2_048
    );
    assert_eq!(two.recovery_value_count, 32);
    assert!(two.assembly_peak_bytes > one.assembly_peak_bytes);

    let mut unmapped = model;
    unmapped.connectors[0].kind = runmat_analysis_core::ConnectorKind::SurfaceNormalFoundation {
        target: runmat_analysis_core::RegionTarget::boundary(
            runmat_meshing_core::fixtures::fixture_face("missing_face"),
        ),
        modulus: runmat_analysis_core::FoundationModulus::new(1.0),
    };
    assert!(matches!(
        connector_buffers::estimate_connector_buffers(
            &unmapped,
            Some(&projection),
            projection.nodes.len() as u64,
        ),
        Err(ResourcePlanningError::Invalid(message)) if message.contains("foundation_a")
    ));

    let mut point_model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    point_model
        .connectors
        .push(runmat_analysis_core::AnalysisConnector {
            connector_id: "tip_spring".into(),
            kind: runmat_analysis_core::ConnectorKind::IsotropicGroundSpring {
                target: runmat_analysis_core::RegionTarget::point("node:2"),
                stiffness: runmat_analysis_core::Stiffness::new(1.0),
            },
        });
    let point = connector_buffers::estimate_connector_buffers(&point_model, None, 2).unwrap();
    let point_metadata = 4_096 + 4 * "tip_spring".len() as u64;
    assert_eq!(point.assembly_peak_bytes, point_metadata);
    assert_eq!(point.recovery_retained_bytes, 2 * point_metadata + 1_024);
    assert_eq!(point.recovery_value_count, 16);
}

#[test]
fn structural_symbolic_plan_is_preassembly_and_material_independent() {
    let (model, mesh, projection, constraints) = structural_fixture();
    let operators =
        StructuralSymbolicOperators::derive(&projection, &constraints, &model, &mesh).unwrap();
    let patterns = operators.patterns();
    let facts = SparseResourceFacts::from_patterns(patterns).unwrap();

    assert_eq!(facts.rows, Count(12));
    assert_eq!(facts.columns, Count(12));
    assert_eq!(
        facts.operator_nonzeros,
        vec![Count(144), Count(48), Count(144)]
    );
    assert_eq!(operators.stiffness_upper_bound().rows(), 12);
    assert_eq!(
        operators.identity(),
        StructuralSymbolicOperators::derive(&projection, &constraints, &model, &mesh)
            .unwrap()
            .identity()
    );
}

#[test]
fn structural_preflight_derives_numerical_facts_before_assembly() {
    let (mut model, mesh, projection, constraints) = structural_fixture();
    model.sequence.steps[0].observations = vec![runmat_analysis_core::ObservationRequest::Field(
        runmat_analysis_core::ObservationSpec {
            observation_id: runmat_analysis_core::ObservationId("whole_displacement".into()),
            quantity: runmat_analysis_core::ObservationQuantity::Displacement,
            value_shape: runmat_analysis_core::ObservationValueShape::Vector3,
            target: runmat_analysis_core::ObservationTarget::WholeModel,
            source_location: runmat_analysis_core::FieldLocation::Node,
            output_location: runmat_analysis_core::FieldLocation::Node,
            recovery: runmat_analysis_core::RecoveryPolicy::Native,
            component: runmat_analysis_core::ObservationComponent::All,
            frame: runmat_analysis_core::CoordinateFrame::Global,
            sign_convention: runmat_analysis_core::SignConvention::Native,
            complex_representation: runmat_analysis_core::ComplexRepresentation::Real,
            sampling: runmat_analysis_core::SamplingPolicy::Final,
            reduction: None,
            storage: runmat_analysis_core::ObservationStoragePolicy {
                precision: runmat_analysis_core::SignalPrecision::Real64,
                compression: runmat_analysis_core::CompressionPolicy::Lossless,
                retention: runmat_analysis_core::RetentionPolicy::Canonical,
                partial_access: runmat_analysis_core::PartialAccessPolicy::ManifestAndChunks,
            },
            purpose: runmat_analysis_core::ObservationPurpose::Publication,
            cost: runmat_analysis_core::ObservationCostPolicy::UpperBound(
                runmat_analysis_core::ObservationCostEstimate {
                    operation_count: 1_000,
                    peak_resident_bytes: 1_000_000,
                    stored_bytes: 1_000_000,
                    transfer_bytes: 1_000_000,
                    collective_count: 0,
                },
            ),
        },
    )];
    let field_definitions = crate::post::fields::structural_observable_field_definitions().unwrap();
    let observations = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraints,
        &field_definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    let workload = crate::execution::StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: crate::contracts::ComputeBackend::Cpu,
        preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        algebra_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };

    let plan = build_structural_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &workload,
        &[],
        LocalExecutionPlan::serial(),
        None,
    )
    .unwrap();
    compiler_phase::assert_carried_request_identity(&workload, &mesh, &observations, &plan);
    let mut serial_workload = workload.clone();
    serial_workload.observation_execution =
        crate::results::observation::ObservationExecutionPolicy::serial();
    let serial_plan = build_structural_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &serial_workload,
        &[],
        LocalExecutionPlan::serial(),
        None,
    )
    .unwrap();
    let direct_plan = build_static_resource_plan(StaticResourcePreflight {
        compilation: None,
        prepared_inputs: None,
        model: &serial_workload.model,
        step: serial_workload.procedure_step().unwrap(),
        backend: serial_workload.backend,
        preconditioner: serial_workload.preconditioner,
        observation_execution: serial_workload.observation_execution,
        mesh: Some(&mesh),
        constraints: &constraints,
        observations: &observations,
        accelerator_requests: &[],
        local_execution: LocalExecutionPlan::serial(),
    })
    .unwrap();
    assert_eq!(direct_plan, serial_plan);
    let solver_model = serial_workload.solver_model().unwrap();
    let solver_projection_plan = build_static_resource_plan(StaticResourcePreflight {
        compilation: None,
        prepared_inputs: None,
        model: &solver_model,
        step: &solver_model.sequence.steps[0],
        backend: serial_workload.backend,
        preconditioner: serial_workload.preconditioner,
        observation_execution: serial_workload.observation_execution,
        mesh: Some(&mesh),
        constraints: &constraints,
        observations: &observations,
        accelerator_requests: &[],
        local_execution: LocalExecutionPlan::serial(),
    })
    .unwrap();
    assert_eq!(
        solver_projection_plan.input.structural_operator().unwrap().solve,
        plan.input.structural_operator().unwrap().solve,
        "the worker's authored workload and FEA's single-step solver projection must admit identical solver work",
    );
    let four_threads = LocalExecutionPlan::fixed(LocalThreadCount::new(4).unwrap()).unwrap();
    let parallel_plan = build_structural_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &workload,
        &[],
        four_threads,
        None,
    )
    .unwrap();

    assert_eq!(plan.request.cpu_millicores, 1_000);
    assert_eq!(parallel_plan.request.cpu_millicores, 4_000);
    assert_eq!(
        parallel_plan.request.scratch_bytes - plan.request.scratch_bytes,
        3 * 128 * 1024
    );
    assert_ne!(parallel_plan.identity, plan.identity);
    let SolveResourcePolicy::Iterative { vector_count, .. } =
        &parallel_plan.input.structural_operator().unwrap().solve
    else {
        panic!("structural preflight must use an iterative solve policy");
    };
    assert_eq!(
        *vector_count,
        Count(9),
        "scoped SpMV merge needs one extra vector"
    );
    let recovery_bound = plan
        .input
        .structural_operator()
        .unwrap()
        .phase_resources
        .resident_value_count
        .0;
    assert!(
        recovery_bound
            >= mesh.topology.nodes.len() as u64 * 7
                + mesh.topology.volume_elements.len() as u64 * 123,
        "recovery admission must cover element fields and their scoped merge",
    );

    assert_eq!(
        plan.input.topology_digest(),
        constraints.topology.discretization_digest()
    );
    assert_eq!(
        plan.input.structural_operator().unwrap().sparse.rows,
        Count(12)
    );
    assert_eq!(
        plan.input
            .structural_operator()
            .unwrap()
            .sparse
            .operator_nonzeros,
        vec![Count(144), Count(48), Count(144)]
    );
    assert_eq!(
        plan.input.structural_operator().unwrap().topology,
        TopologyResourceFacts::from_projection(&projection).unwrap()
    );
    assert_eq!(
        plan.input.structural_operator().unwrap().constraints,
        ConstraintResourceFacts::from_plan(&constraints).unwrap()
    );
    let mut foundation_workload = workload.clone();
    foundation_workload
        .model
        .connectors
        .push(runmat_analysis_core::AnalysisConnector {
            connector_id: "preflight_foundation".into(),
            kind: runmat_analysis_core::ConnectorKind::SurfaceNormalFoundation {
                target: runmat_analysis_core::RegionTarget::boundary(
                    runmat_meshing_core::fixtures::fixture_face("root"),
                ),
                modulus: runmat_analysis_core::FoundationModulus::new(1.0),
            },
        });
    let foundation_plan = build_structural_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &foundation_workload,
        &[],
        LocalExecutionPlan::serial(),
        None,
    )
    .unwrap();
    let foundation_bytes = connector_buffers::estimate_connector_buffers(
        &foundation_workload.model,
        Some(&projection),
        projection.nodes.len() as u64,
    )
    .unwrap();
    assert!(foundation_bytes.assembly_peak_bytes > 0);
    assert_eq!(
        foundation_plan
            .input
            .structural_operator()
            .unwrap()
            .assembly
            .contribution_bytes_upper_bound
            .0,
        plan.input
            .structural_operator()
            .unwrap()
            .assembly
            .contribution_bytes_upper_bound
            .0
            + foundation_bytes.assembly_peak_bytes,
    );
    let baseline_operator = plan.input.structural_operator().unwrap();
    let foundation_operator = foundation_plan.input.structural_operator().unwrap();
    let connector_payload_bytes =
        foundation_bytes.recovery_value_count * 8 + foundation_bytes.recovery_retained_bytes;
    assert_eq!(
        foundation_operator.phase_resources.resident_value_count.0,
        baseline_operator.phase_resources.resident_value_count.0
            + foundation_bytes.recovery_value_count,
    );
    assert_eq!(
        foundation_operator.phase_resources.retained_bytes.0,
        baseline_operator.phase_resources.retained_bytes.0 + connector_payload_bytes,
    );
    assert_eq!(
        foundation_operator.publication.result_artifact_bytes.0,
        baseline_operator.publication.result_artifact_bytes.0 + connector_payload_bytes,
    );
    assert_ne!(foundation_plan.identity, plan.identity);
    assert_eq!(
        plan.input
            .structural_operator()
            .unwrap()
            .field_recovery
            .requirement_count,
        Count(1)
    );
    assert!(plan.breakdown.field_recovery_arithmetic_operation_count.0 > 0);
    assert!(
        plan.input.observations().unwrap().transfer_bytes
            > serial_plan.input.observations().unwrap().transfer_bytes
    );
    assert_ne!(plan.identity, serial_plan.identity);
    let operators =
        StructuralSymbolicOperators::derive(&projection, &constraints, &workload.model, &mesh)
            .unwrap();
    let expected_admission =
        crate::algebra::CoercivityWorkEnvelope::derive(operators.stiffness_upper_bound()).unwrap();
    let SolveResourcePolicy::Iterative { admission, .. } =
        &plan.input.structural_operator().unwrap().solve
    else {
        panic!("structural preflight must authorize iterative solver admission");
    };
    assert_eq!(admission.dimension, Count(expected_admission.dimension()));
    assert_eq!(
        admission.factor_nonzero_upper_bound,
        Count(expected_admission.factor_nonzero_limit())
    );
    // The static solve is proved by its Cholesky factor: only factorization work and the
    // factor's resident payload are reserved, never the local-energy proof's graph budgets.
    assert_eq!(
        admission.update_count_upper_bound,
        Count(expected_admission.factor_flop_limit())
    );
    assert_eq!(
        admission.resident_bytes_upper_bound,
        ByteCount(
            (expected_admission.factor_nonzero_limit() * 16
                + (expected_admission.dimension() + 1) * 8)
                .max(expected_admission.factor_resident_limit())
        )
    );
    plan.validate_identity().unwrap();

    let mut gpu_workload = workload;
    gpu_workload.backend = crate::contracts::ComputeBackend::Gpu;
    gpu_workload.algebra_backend =
        crate::solve::backend::kind::LinearAlgebraBackendKind::Accelerator;
    assert!(build_structural_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &gpu_workload,
        &[],
        LocalExecutionPlan::serial(),
        None,
    )
    .is_err());
    let accelerator = AcceleratorRequest {
        class: runmat_execution::resource::AcceleratorClass::new("test-gpu").unwrap(),
        count: 1,
        minimum_allocation_bytes: 1024 * 1024 * 1024,
        provider: None,
        required_features: Default::default(),
    };
    let accelerated = build_structural_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &gpu_workload,
        std::slice::from_ref(&accelerator),
        LocalExecutionPlan::serial(),
        None,
    )
    .unwrap();
    assert_eq!(accelerated.request.accelerators, vec![accelerator]);
    assert!(accelerated
        .request
        .required_capabilities
        .contains(&Capability::Accelerator(
            runmat_execution::resource::AcceleratorClass::new("test-gpu").unwrap()
        )));
}

#[test]
fn mixed_preflight_binds_complete_topology_and_connector_sparsity() {
    let (model, mesh, projection, constraints) = mixed_structural_fixture();
    let observations = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraints,
        &[],
        ObservationLimits::default(),
    )
    .unwrap();
    let workload = crate::execution::StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model: model.clone(),
        backend: crate::contracts::ComputeBackend::Cpu,
        preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        algebra_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let plan = build_structural_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &workload,
        &[],
        LocalExecutionPlan::serial(),
        None,
    )
    .unwrap();

    assert_eq!(
        plan.input.structural_operator().unwrap().topology.nodes,
        Count(5)
    );
    assert_eq!(
        plan.input.structural_operator().unwrap().topology.elements,
        Count(2)
    );
    assert_eq!(
        plan.input.structural_operator().unwrap().sparse.rows,
        Count(21)
    );
    assert_eq!(
        plan.input.topology_digest(),
        constraints.topology.discretization_digest()
    );
    assert_eq!(
        plan.identity,
        build_structural_resource_plan(
            Some(&mesh),
            &constraints,
            &observations,
            &workload,
            &[],
            LocalExecutionPlan::serial(),
            None,
        )
        .unwrap()
        .identity
    );

    let with_connector =
        StructuralSymbolicOperators::derive(&projection, &constraints, &model, &mesh).unwrap();
    let mut without_connector_model = model;
    without_connector_model.connectors.clear();
    let without_connector = StructuralSymbolicOperators::derive(
        &projection,
        &constraints,
        &without_connector_model,
        &mesh,
    )
    .unwrap();
    assert!(
        with_connector
            .stiffness_upper_bound()
            .column_indices()
            .len()
            > without_connector
                .stiffness_upper_bound()
                .column_indices()
                .len()
    );
    plan.validate_identity().unwrap();
}

#[test]
fn solid_consistent_mass_and_rayleigh_damping_reserve_element_couplings() {
    let (mut model, mesh, projection, constraints) = structural_fixture();
    let baseline =
        StructuralSymbolicOperators::derive(&projection, &constraints, &model, &mesh).unwrap();
    let first = mesh.topology.nodes[0].node_id;
    let second = mesh.topology.nodes[1].node_id;
    model
        .connectors
        .push(runmat_analysis_core::AnalysisConnector {
            connector_id: "solid_axial_dashpot".into(),
            kind: runmat_analysis_core::ConnectorKind::AxialDashpot {
                primary: runmat_analysis_core::RegionTarget::point(format!("node:{first}")),
                secondary: runmat_analysis_core::RegionTarget::point(format!("node:{second}")),
                damping: runmat_analysis_core::Damping::new(100.0),
            },
        });
    let with_connector =
        StructuralSymbolicOperators::derive(&projection, &constraints, &model, &mesh).unwrap();
    assert_eq!(baseline.mass_upper_bound().column_indices().len(), 48);
    assert_eq!(
        baseline
            .mass_upper_bound()
            .row_range(0)
            .map(|entry| baseline.mass_upper_bound().column(entry))
            .collect::<Vec<_>>(),
        vec![0, 3, 6, 9],
    );
    assert_eq!(
        baseline.damping_upper_bound().column_indices(),
        baseline.stiffness_upper_bound().column_indices(),
    );
    assert_eq!(
        with_connector.stiffness_upper_bound().column_indices(),
        baseline.stiffness_upper_bound().column_indices(),
    );
    assert_eq!(
        with_connector.damping_upper_bound().column_indices(),
        with_connector.stiffness_upper_bound().column_indices(),
    );
    assert_eq!(with_connector.identity(), baseline.identity());
}

#[test]
fn explicit_preflight_derives_complete_bounds_without_a_solver_mesh() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(&model, None).unwrap();
    let observations = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        None,
        &constraints,
        &[],
        ObservationLimits::default(),
    )
    .unwrap();
    let workload = crate::execution::StructuralLinearWorkload {
        schema_version: 2,
        step_id: model.sequence.steps[0].step_id.clone(),
        model: model.clone(),
        backend: crate::contracts::ComputeBackend::Cpu,
        preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        algebra_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        state_input: None,
    };
    let plan = build_structural_resource_plan(
        None,
        &constraints,
        &observations,
        &workload,
        &[],
        LocalExecutionPlan::serial(),
        None,
    )
    .unwrap();
    let structural = model.structural.as_ref().unwrap();

    assert_eq!(
        plan.input.topology_digest(),
        constraints.topology.discretization_digest()
    );
    assert_eq!(
        plan.input.structural_operator().unwrap().topology.nodes,
        Count(structural.nodes.len() as u64)
    );
    assert_eq!(
        plan.input.structural_operator().unwrap().topology.elements,
        Count(structural.elements.len() as u64)
    );
    assert_eq!(
        plan.input
            .structural_operator()
            .unwrap()
            .topology
            .boundary_faces,
        Count(0)
    );
    assert_eq!(
        plan.input.structural_operator().unwrap().sparse.rows,
        Count((structural.nodes.len() * 6) as u64)
    );
    assert_eq!(
        plan.identity,
        build_structural_resource_plan(
            None,
            &constraints,
            &observations,
            &workload,
            &[],
            LocalExecutionPlan::serial(),
            None,
        )
        .unwrap()
        .identity
    );
}

#[test]
fn exact_small_estimate_separates_all_resource_classes() {
    let plan = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let catalog = plan
        .input
        .demand
        .output_facts(crate::execution::AnalysisNumericalObjectKind::StatePublicationProfiles)
        .unwrap();

    assert_eq!(plan.breakdown.topology_bytes, ByteCount(1_088));
    assert_eq!(plan.breakdown.csr_row_offset_bytes, ByteCount(120));
    assert_eq!(plan.breakdown.csr_column_index_bytes, ByteCount(144));
    assert_eq!(plan.breakdown.csr_value_bytes, ByteCount(144));
    assert_eq!(plan.breakdown.assembly_contribution_bytes, ByteCount(240));
    assert_eq!(plan.breakdown.workspace_bytes, ByteCount(160));
    assert_eq!(plan.breakdown.preconditioner_bytes, ByteCount(32));
    assert_eq!(plan.breakdown.factorization_bytes, ByteCount(200));
    assert_eq!(plan.breakdown.constraint_bytes, ByteCount(120));
    assert_eq!(plan.breakdown.observation_operation_count, Count(100));
    assert_eq!(
        plan.breakdown.observation_maximum_samples_per_observation,
        Count(1)
    );
    assert_eq!(plan.breakdown.observation_produced_sample_count, Count(1));
    assert_eq!(plan.breakdown.observation_retained_sample_count, Count(1));
    assert_eq!(
        plan.request.memory_bytes,
        plan.breakdown.worker_overhead_bytes.0
            + plan.breakdown.input_transport_closure_bytes.0
            + plan.breakdown.resource_plan_transport_closure_bytes.0
            + plan.breakdown.input_decoded_resident_bytes.0
            + plan.breakdown.resource_plan_decoded_resident_bytes.0
            + plan.breakdown.topology_bytes.0
            + plan.breakdown.csr_row_offset_bytes.0
            + plan.breakdown.csr_column_index_bytes.0
            + plan.breakdown.csr_value_bytes.0
            + plan.breakdown.constraint_bytes.0
            + plan.breakdown.assembly_contribution_bytes.0.max(
                (plan.breakdown.workspace_bytes.0
                    + plan.breakdown.preconditioner_bytes.0
                    + plan.breakdown.factorization_bytes.0)
                    .max(
                        plan.breakdown.phase_resident_bytes.0
                            + catalog.construction_resident_bytes_admission_budget.0
                            + plan.breakdown.observation_peak_bytes.0
                            + plan.breakdown.output_set_bytes.0,
                    ),
            )
    );
    assert_eq!(plan.request.scratch_bytes, 45);
    assert_eq!(plan.breakdown.result_artifact_bytes, ByteCount(34));
    assert_eq!(plan.breakdown.phase_retained_bytes, ByteCount(32));
    assert_eq!(plan.request.max_artifact_bytes, 304);
    assert_eq!(plan.request.max_egress_bytes, 221);
    assert_eq!(
        plan.request.max_relay_bytes,
        12 + plan.breakdown.resource_plan_transport_closure_bytes.0
    );
    plan.validate_identity().unwrap();
}

#[test]
fn overflow_and_impossible_sparse_facts_are_rejected() {
    let mut overflow = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut overflow).sparse.rows = Count(u64::MAX);
    operator_mut(&mut overflow).sparse.columns = Count(u64::MAX);
    assert!(matches!(
        build_resource_plan(overflow),
        Err(ResourcePlanningError::Overflow("sparse dense bound"))
    ));

    let mut impossible = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut impossible).sparse.operator_nonzeros[0] = Count(17);
    assert!(matches!(
        build_resource_plan(impossible),
        Err(ResourcePlanningError::Invalid(_))
    ));
}

#[test]
fn understated_resident_bounds_are_rejected() {
    let mut topology = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut topology)
        .topology
        .estimated_resident_bytes = ByteCount(1);
    assert!(matches!(
        build_resource_plan(topology),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut constraints = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut constraints)
        .constraints
        .estimated_resident_bytes = ByteCount(1);
    assert!(matches!(
        build_resource_plan(constraints),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut contributions = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut contributions)
        .assembly
        .contribution_bytes_upper_bound = ByteCount(1);
    assert!(matches!(
        build_resource_plan(contributions),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut recovery = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut recovery)
        .phase_resources
        .resident_bytes_upper_bound = ByteCount(1);
    assert!(matches!(
        build_resource_plan(recovery),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut preconditioner = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut preconditioner).solve = SolveResourcePolicy::Iterative {
        vector_count: Count(2),
        scalar: NumericalScalar::RealF64,
        preconditioner: PreconditionerStorage::SparseCsr {
            nonzero_upper_bound: Count(10),
            resident_bytes_upper_bound: ByteCount(1),
        },
        transpose_index_bytes_upper_bound: ByteCount(0),
        path: StaticSolvePath::ProvingFactorization,
        admission: SolverAdmissionResourcePolicy {
            dimension: Count(4),
            factor_nonzero_upper_bound: Count(10),
            update_count_upper_bound: Count(12),
            resident_bytes_upper_bound: ByteCount(200),
        },
    };
    assert!(matches!(
        build_resource_plan(preconditioner),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut factor = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut factor).solve = SolveResourcePolicy::DirectFactorization {
        vector_count: Count(2),
        scalar: NumericalScalar::RealF64,
        factor_nonzero_upper_bound: Count(10),
        factor_bytes_upper_bound: ByteCount(1),
    };
    assert!(matches!(
        build_resource_plan(factor),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut admission = input(PartitioningPolicy::Monolithic);
    let SolveResourcePolicy::Iterative {
        admission: policy, ..
    } = &mut operator_mut(&mut admission).solve
    else {
        unreachable!()
    };
    policy.resident_bytes_upper_bound = ByteCount(1);
    assert!(matches!(
        build_resource_plan(admission),
        Err(ResourcePlanningError::Invalid(_))
    ));
}

#[test]
fn complex_solve_vectors_and_diagonal_storage_use_complex_width() {
    let mut real_input = input(PartitioningPolicy::Monolithic);
    // Keep this width test in the solve-dominant regime rather than the independently
    // reserved producer-catalog construction phase.
    let SolveResourcePolicy::Iterative { vector_count, .. } =
        &mut operator_mut(&mut real_input).solve
    else {
        panic!("iterative fixture")
    };
    *vector_count = Count(10_000);
    let real = build_resource_plan(real_input.clone()).unwrap();
    let mut complex_input = real_input;
    operator_mut(&mut complex_input).solve = SolveResourcePolicy::Iterative {
        vector_count: Count(10_000),
        scalar: NumericalScalar::ComplexF64,
        preconditioner: PreconditionerStorage::Diagonal,
        transpose_index_bytes_upper_bound: ByteCount(0),
        path: StaticSolvePath::ProvingFactorization,
        admission: SolverAdmissionResourcePolicy {
            dimension: Count(4),
            factor_nonzero_upper_bound: Count(10),
            update_count_upper_bound: Count(12),
            resident_bytes_upper_bound: ByteCount(280),
        },
    };
    let complex = build_resource_plan(complex_input).unwrap();

    assert_eq!(real.breakdown.workspace_bytes, ByteCount(320_000));
    assert_eq!(complex.breakdown.workspace_bytes, ByteCount(640_000));
    assert_eq!(complex.breakdown.preconditioner_bytes, ByteCount(64));
    assert_eq!(
        complex.request.memory_bytes - real.request.memory_bytes,
        (640_000 + 64 + 280) - (320_000 + 32 + 200)
    );
}

#[test]
fn explicit_frequency_workspaces_require_valid_dimensions_and_bounds() {
    let mut gmres = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut gmres).solve = SolveResourcePolicy::RestartedGmres {
        block_dimension: Count(8),
        restart: Count(9),
        maximum_iterations: Count(16),
        workspace_bytes_upper_bound: ByteCount(1_024),
    };
    assert!(matches!(
        build_resource_plan(gmres),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut modal = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut modal).solve = SolveResourcePolicy::ModalDenseHarmonic {
        full_dimension: Count(4),
        mode_count: Count(5),
        workspace_bytes_upper_bound: ByteCount(1_024),
    };
    assert!(matches!(
        build_resource_plan(modal),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut valid = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut valid).solve = SolveResourcePolicy::RestartedGmres {
        block_dimension: Count(8),
        restart: Count(4),
        maximum_iterations: Count(16),
        workspace_bytes_upper_bound: ByteCount(1_024),
    };
    let plan = build_resource_plan(valid).unwrap();
    assert_eq!(plan.breakdown.workspace_bytes, ByteCount(1_024));
}

#[test]
fn result_artifacts_are_accounted_independently_from_recovery_retention() {
    let baseline = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let mut retained = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut retained).phase_resources.retained_bytes = ByteCount(99);
    let retained = build_resource_plan(retained).unwrap();
    assert_eq!(
        retained.request.max_artifact_bytes,
        baseline.request.max_artifact_bytes
    );

    let mut result = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut result).publication.result_artifact_bytes = ByteCount(100);
    let result = build_resource_plan(result).unwrap();
    assert!(result.request.max_artifact_bytes > baseline.request.max_artifact_bytes);
}

#[test]
fn accuracy_ceiling_is_enforced_before_inventory_admission() {
    let mut memory = input(PartitioningPolicy::Monolithic);
    memory.accuracy_ceiling.maximum_resident_bytes = Some(1_000);
    assert!(matches!(
        admit_resource_plan(memory, &inventory(10_000, 10_000)),
        ResourceAdmission::Reject {
            reason: ResourcePlanningError::AccuracyCeiling {
                resource: "resident memory bytes",
                required: _,
                maximum: 1_000,
            }
        }
    ));

    let mut samples = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut samples).accuracy_demand.samples = Count(5);
    assert!(matches!(
        build_resource_plan(samples),
        Err(ResourcePlanningError::AccuracyCeiling {
            resource: "samples",
            required: 5,
            maximum: 4,
        })
    ));
}

#[test]
fn inventory_admits_or_rejects_a_monolithic_plan() {
    let plan = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    assert!(matches!(
        admit_resource_plan(
            input(PartitioningPolicy::Monolithic),
            &inventory(plan.request.memory_bytes, 100)
        ),
        ResourceAdmission::Admit { .. }
    ));
    assert!(matches!(
        admit_resource_plan(
            input(PartitioningPolicy::Monolithic),
            &inventory(plan.request.memory_bytes - 1, 100)
        ),
        ResourceAdmission::Reject {
            reason: ResourcePlanningError::MemoryBudget {
                required_memory_bytes,
                available_memory_bytes,
                ..
            }
        } if required_memory_bytes == plan.request.memory_bytes
            && available_memory_bytes == plan.request.memory_bytes - 1
    ));
}

// G25: a 290,919,200-value retained basis (the drone arm modal study's 1,135,827 DOFs at the
// 256-vector search ceiling, its size before G26 sized the subspace by the request). No fixed
// item or byte constant refuses it; the worker's admitted memory and scratch decide, and a
// refusal names the bytes needed and the bytes available.
#[test]
fn modal_basis_above_one_hundred_million_values_plans_within_the_admitted_budget() {
    let (full_dofs, modes, nodes) = (1_135_827_u64, 256_u64, 378_609_u64);
    let values = full_dofs * modes + 2 * modes * modes + 64 * modes + 32;
    assert_eq!(values, 290_919_200);
    let output = modal::output::result_output(values, modes, 0, full_dofs, nodes).unwrap();
    assert!(output.encoded_payload_bytes_upper_bound.0 > 1024 * 1024 * 1024);

    let arm_input = || {
        let mut input = input(PartitioningPolicy::Monolithic);
        input.accuracy_ceiling.maximum_resident_bytes = None;
        let FeaResourceDemand::StructuralLinearPipeline(demand) = &mut input.demand else {
            unreachable!()
        };
        demand.state_profiles_output = output.clone();
        input
    };
    let plan = build_resource_plan(arm_input()).unwrap();
    assert!(plan.request.memory_bytes > output.encoded_payload_bytes_upper_bound.0);
    let scratch = plan.request.scratch_bytes;
    assert!(matches!(
        admit_resource_plan(arm_input(), &inventory(plan.request.memory_bytes, scratch)),
        ResourceAdmission::Admit { .. }
    ));
    let available = 1024 * 1024 * 1024;
    match admit_resource_plan(arm_input(), &inventory(available, scratch)) {
        ResourceAdmission::Reject {
            reason:
                reason @ ResourcePlanningError::MemoryBudget {
                    required_memory_bytes,
                    available_memory_bytes,
                    required_scratch_bytes,
                    available_scratch_bytes,
                },
        } => {
            assert_eq!(required_memory_bytes, plan.request.memory_bytes);
            assert_eq!(available_memory_bytes, available);
            assert_eq!(required_scratch_bytes, scratch);
            assert_eq!(available_scratch_bytes, scratch);
            let message = reason.to_string();
            assert!(message.contains(&required_memory_bytes.to_string()));
            assert!(message.contains(&available.to_string()));
        }
        other => panic!("expected a memory-budget refusal, got {other:?}"),
    }
}

#[test]
fn operator_domain_work_gets_a_conservative_partition_recommendation() {
    let partitioning = PartitioningPolicy::OperatorDomain {
        maximum_partitions: 4,
        replicated_memory_bytes_per_partition: ByteCount(20),
        halo_memory_bytes_per_partition: ByteCount(10),
        replicated_scratch_bytes_per_partition: ByteCount(2),
        halo_scratch_bytes_per_partition: ByteCount(3),
    };
    let full = build_resource_plan(input(partitioning)).unwrap();
    let available_memory = full.breakdown.fixed_memory_bytes.0
        + full.breakdown.partitionable_memory_bytes.0.div_ceil(3)
        + 30;
    let decision = admit_resource_plan(input(partitioning), &inventory(available_memory, 100));
    let ResourceAdmission::RequiresPartitioning {
        minimum_partition_count,
        per_partition_request,
        ..
    } = decision
    else {
        panic!("operator/domain work should receive a partition recommendation");
    };
    assert_eq!(minimum_partition_count, 3);
    assert!(per_partition_request.memory_bytes <= available_memory);
    assert_eq!(per_partition_request.scratch_bytes, 24);
}

#[test]
fn missing_capability_or_accelerator_rejects_before_memory_admission() {
    let no_capability = ResourceInventory {
        capabilities: BTreeSet::new(),
        ..inventory(64 * 1024 * 1024, 10_000)
    };
    assert!(matches!(
        admit_resource_plan(input(PartitioningPolicy::Monolithic), &no_capability),
        ResourceAdmission::Reject {
            reason: ResourcePlanningError::Capability(Capability::NetworkDenied)
        }
    ));

    let mut accelerated = input(PartitioningPolicy::Monolithic);
    accelerated.worker.accelerators = vec![AcceleratorRequirement {
        class: runmat_execution::resource::AcceleratorClass::new("gpu").unwrap(),
        count: 1,
        memory_bytes_each: ByteCount(4_096),
    }];
    let inadequate = ResourceInventory {
        accelerators: vec![test_accelerator_device("gpu", 2_048)],
        ..inventory(64 * 1024 * 1024, 10_000)
    };
    assert!(matches!(
        admit_resource_plan(accelerated, &inadequate),
        ResourceAdmission::Reject {
            reason: ResourcePlanningError::Accelerator { .. }
        }
    ));
}

#[test]
fn observation_cost_changes_memory_artifacts_transfer_and_identity() {
    let baseline = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let mut changed_input = input(PartitioningPolicy::Monolithic);
    operator_mut(&mut changed_input)
        .observations
        .peak_resident_bytes = ByteCount(400);
    operator_mut(&mut changed_input).observations.stored_bytes = ByteCount(500);
    operator_mut(&mut changed_input).observations.transfer_bytes = ByteCount(450);
    let changed = build_resource_plan(changed_input).unwrap();

    assert!(changed.request.memory_bytes > baseline.request.memory_bytes);
    assert!(changed.request.max_artifact_bytes > baseline.request.max_artifact_bytes);
    assert!(changed.request.max_egress_bytes > baseline.request.max_egress_bytes);
    assert_ne!(changed.identity, baseline.identity);
    assert_eq!(
        build_resource_plan(input(PartitioningPolicy::Monolithic))
            .unwrap()
            .identity,
        baseline.identity
    );
}

#[test]
fn reconciliation_reports_every_overrun_without_mutating_the_plan() {
    let plan = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let identity = plan.identity;
    let actual = AnalysisResourceUsage {
        scheduler: ResourceUsage {
            cpu_core_millis: 1_001,
            memory_byte_millis: u128::from(plan.request.memory_bytes) * 1_000 + 1,
            wall_millis: 1_001,
            egress_bytes: plan.request.max_egress_bytes + 1,
            relay_bytes: plan.request.max_relay_bytes + 1,
            retained_byte_millis: u128::from(plan.request.max_artifact_bytes) * 1_000 + 1,
            ..ResourceUsage::default()
        },
        peak_cpu_millicores: 1_001,
        peak_memory_bytes: ByteCount(plan.request.memory_bytes + 1),
        peak_scratch_bytes: ByteCount(plan.request.scratch_bytes + 1),
        artifact_bytes: ByteCount(plan.request.max_artifact_bytes + 1),
    };
    let overruns = reconcile_resource_usage(&plan, &actual).unwrap();

    assert_eq!(overruns.len(), 10);
    assert_eq!(plan.identity, identity);
    plan.validate_identity().unwrap();
}

#[test]
fn reconciliation_checks_accelerator_device_time() {
    let mut accelerated = input(PartitioningPolicy::Monolithic);
    accelerated.worker.accelerators = vec![AcceleratorRequirement {
        class: runmat_execution::resource::AcceleratorClass::new("gpu").unwrap(),
        count: 1,
        memory_bytes_each: ByteCount(4_096),
    }];
    let plan = build_resource_plan(accelerated).unwrap();
    let actual = AnalysisResourceUsage {
        scheduler: ResourceUsage {
            accelerator_millis: vec![{
                let device = test_accelerator_device("gpu", 4_096);
                runmat_execution::resource::AcceleratorUsage {
                    class: device.class,
                    provider: device.provider.id,
                    device_id: device.id,
                    device_millis: 1_001,
                }
            }],
            ..ResourceUsage::default()
        },
        peak_cpu_millicores: 0,
        peak_memory_bytes: ByteCount(0),
        peak_scratch_bytes: ByteCount(0),
        artifact_bytes: ByteCount(0),
    };

    assert!(matches!(
        reconcile_resource_usage(&plan, &actual).unwrap().as_slice(),
        [ResourceOverrun::AcceleratorTime { .. }]
    ));
}

#[test]
fn resource_usage_requires_the_exact_planned_accelerator_reservation() {
    let mut accelerated = input(PartitioningPolicy::Monolithic);
    accelerated.worker.accelerators = vec![AcceleratorRequirement {
        class: runmat_execution::resource::AcceleratorClass::new("gpu").unwrap(),
        count: 1,
        memory_bytes_each: ByteCount(4_096),
    }];
    let plan = build_resource_plan(accelerated).unwrap();
    let attempt_id = runmat_execution::identity::AttemptId::derive(&[b"accelerated-attempt"]);
    let worker_id = runmat_execution::identity::WorkerId::derive(&[b"accelerated-worker"]);
    // Each requested accelerator is accounted against a leased device that serves it.
    let record = |allocation: runmat_execution::resource::ResourceRequest| {
        let assignment = runmat_execution::resource::ResourceAssignment {
            accelerator_leases: allocation
                .accelerators
                .iter()
                .map(|request| {
                    runmat_execution::resource::AcceleratorDeviceLease::for_attempt(
                        attempt_id,
                        1,
                        &test_accelerator_device(request.class.as_str(), 4_096),
                    )
                    .unwrap()
                })
                .collect(),
        };
        ResourceUsageAttempt {
            attempt_id,
            worker_id,
            plan: plan.identity,
            resources: runmat_execution::resource::ResourceAccounting::for_assigned_interval(
                allocation,
                &assignment,
                1,
                0,
                0,
                0,
            )
            .unwrap(),
        }
    };
    let record_usage = |allocation| {
        record_resource_usage(
            &plan,
            Digest::sha256("accelerated-workload"),
            Digest::sha256("accelerated-result"),
            vec![record(allocation)],
            1,
        )
    };

    assert!(record_usage(plan.request.clone()).is_ok());

    let mut missing = plan.request.clone();
    missing.accelerators.clear();
    assert!(matches!(
        record_usage(missing),
        Err(ResourcePlanningError::Invalid(message))
            if message == "resource usage attempt allocation differs from its plan"
    ));

    let mut understated = plan.request.clone();
    understated.accelerators[0].minimum_allocation_bytes -= 1;
    assert!(matches!(
        record_usage(understated),
        Err(ResourcePlanningError::Invalid(message))
            if message == "resource usage attempt allocation differs from its plan"
    ));

    let mut duplicated = plan.request.clone();
    duplicated
        .accelerators
        .push(duplicated.accelerators[0].clone());
    // Duplicate accelerator requests cannot be accounted, whichever check sees them first.
    assert!(
        runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            duplicated, 1, 0, 0, 0
        )
        .is_err()
    );
}

#[test]
fn resource_usage_evidence_binds_plan_workload_result_attempt_and_worker() {
    let plan = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let allocation = plan.request.clone();
    let attempt_id = runmat_execution::identity::AttemptId::derive(&[b"attempt"]);
    let worker_id = runmat_execution::identity::WorkerId::derive(&[b"worker"]);
    let attempt = ResourceUsageAttempt {
        attempt_id,
        worker_id,
        plan: plan.identity,
        resources: runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            allocation.clone(),
            1,
            0,
            0,
            0,
        )
        .unwrap(),
    };
    let retry_attempt = ResourceUsageAttempt {
        attempt_id: runmat_execution::identity::AttemptId::derive(&[b"retry-attempt"]),
        worker_id: runmat_execution::identity::WorkerId::derive(&[b"retry-worker"]),
        plan: plan.identity,
        resources: runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            allocation.clone(),
            1,
            0,
            0,
            0,
        )
        .unwrap(),
    };
    let evidence = record_resource_usage(
        &plan,
        Digest::sha256("workload"),
        Digest::sha256("result"),
        vec![retry_attempt, attempt],
        2,
    )
    .unwrap();
    assert!(evidence.overruns.is_empty());
    assert_eq!(
        ResourceUsageEvidence::from_canonical_bytes(&evidence.canonical_bytes().unwrap()).unwrap(),
        evidence
    );
    assert_eq!(evidence.attempts.len(), 2);
    assert!(evidence
        .attempts
        .windows(2)
        .all(|pair| pair[0].attempt_id < pair[1].attempt_id));

    let mut tampered = serde_json::to_value(&evidence).unwrap();
    tampered["actual"]["scheduler"]["cpu_core_millis"] = serde_json::json!(0);
    assert!(serde_json::from_value::<ResourceUsageEvidence>(tampered).is_err());

    let mut substituted = allocation;
    substituted.memory_bytes += 1;
    let substituted_attempt = ResourceUsageAttempt {
        attempt_id,
        worker_id,
        plan: plan.identity,
        resources: runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            substituted,
            1,
            0,
            0,
            0,
        )
        .unwrap(),
    };
    assert!(matches!(
        record_resource_usage(
            &plan,
            Digest::sha256("workload"),
            Digest::sha256("result"),
            vec![substituted_attempt],
            1,
        ),
        Err(ResourcePlanningError::Invalid(message))
            if message == "resource usage attempt allocation differs from its plan"
    ));
}

#[test]
fn resource_usage_rejects_under_reserved_attempts_even_when_they_fit_the_plan_budget() {
    let plan = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let attempt_id = runmat_execution::identity::AttemptId::derive(&[b"under-reserved-attempt"]);
    let worker_id = runmat_execution::identity::WorkerId::derive(&[b"under-reserved-worker"]);
    let record_usage = |allocation| {
        record_resource_usage(
            &plan,
            Digest::sha256("under-reserved-workload"),
            Digest::sha256("under-reserved-result"),
            vec![ResourceUsageAttempt {
                attempt_id,
                worker_id,
                plan: plan.identity,
                resources: runmat_execution::resource::ResourceAccounting::for_reserved_interval(
                    allocation, 1, 0, 0, 0,
                )
                .unwrap(),
            }],
            1,
        )
    };

    assert!(record_usage(plan.request.clone()).is_ok());
    let mut under_reserved = plan.request.clone();
    under_reserved.memory_bytes -= 1;
    assert!(matches!(
        record_usage(under_reserved),
        Err(ResourcePlanningError::Invalid(message))
            if message == "resource usage attempt allocation differs from its plan"
    ));

    let mut shorter_deadline = plan.request.clone();
    shorter_deadline.max_wall_millis -= 1;
    assert!(matches!(
        record_usage(shorter_deadline),
        Err(ResourcePlanningError::Invalid(message))
            if message == "resource usage attempt allocation differs from its plan"
    ));
}

#[test]
fn resource_usage_evidence_reconciles_each_attempt_against_its_authenticated_plan() {
    let primary = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let mut phase_input = input(PartitioningPolicy::Monolithic);
    phase_input.worker.cpu_millicores = 500;
    phase_input.worker.maximum_wall_milliseconds = 10;
    let phase = build_resource_plan(phase_input).unwrap();
    let phase_attempt_id = runmat_execution::identity::AttemptId::derive(&[b"phase-attempt"]);
    let phase_attempt = ResourceUsageAttempt {
        attempt_id: phase_attempt_id,
        worker_id: runmat_execution::identity::WorkerId::derive(&[b"phase-worker"]),
        plan: phase.identity,
        resources: runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            phase.request.clone(),
            11,
            0,
            0,
            0,
        )
        .unwrap(),
    };
    let primary_attempt = ResourceUsageAttempt {
        attempt_id: runmat_execution::identity::AttemptId::derive(&[b"primary-attempt"]),
        worker_id: runmat_execution::identity::WorkerId::derive(&[b"primary-worker"]),
        plan: primary.identity,
        resources: runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            primary.request.clone(),
            1,
            0,
            0,
            0,
        )
        .unwrap(),
    };

    let evidence = record_resource_graph_usage(
        &primary,
        vec![phase.clone()],
        Digest::sha256("workload"),
        Digest::sha256("result"),
        vec![phase_attempt.clone(), primary_attempt.clone()],
        11,
    )
    .unwrap();
    assert_eq!(evidence.primary_plan, primary.identity);
    assert_eq!(evidence.plans.len(), 2);
    assert!(evidence.plans.contains(&phase));
    assert!(evidence.plans.contains(&primary));
    assert!(matches!(
        evidence.overruns.as_slice(),
        [ResourceUsageAttemptOverrun {
            attempt_id,
            overrun: ResourceOverrun::CpuTime { .. }
        }, ResourceUsageAttemptOverrun {
            attempt_id: memory_attempt_id,
            overrun: ResourceOverrun::MemoryTime { .. }
        }, ResourceUsageAttemptOverrun {
            attempt_id: final_attempt_id,
            overrun: ResourceOverrun::WallTime { .. }
        }]
            if *attempt_id == phase_attempt_id
                && *memory_attempt_id == phase_attempt_id
                && *final_attempt_id == phase_attempt_id
    ));
    assert_eq!(
        ResourceUsageEvidence::from_canonical_bytes(&evidence.canonical_bytes().unwrap()).unwrap(),
        evidence
    );
    let mut substituted_plan_binding = evidence.clone();
    substituted_plan_binding
        .attempts
        .iter_mut()
        .find(|attempt| attempt.plan == phase.identity)
        .unwrap()
        .plan = primary.identity;
    assert!(substituted_plan_binding.validate_identity().is_err());
    let mut altered_catalog = evidence.clone();
    altered_catalog.plans[0].request.memory_bytes = 1;
    assert!(altered_catalog.validate_identity().is_err());

    assert!(matches!(
        record_resource_usage(
            &primary,
            Digest::sha256("workload"),
            Digest::sha256("result"),
            vec![phase_attempt],
            11,
        ),
        Err(ResourcePlanningError::Invalid(message))
            if message == "resource usage attempt refers to an absent resource plan"
    ));

    let substituted = ResourceUsageAttempt {
        attempt_id: phase_attempt_id,
        worker_id: runmat_execution::identity::WorkerId::derive(&[b"phase-worker"]),
        plan: phase.identity,
        resources: runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            primary.request.clone(),
            1,
            0,
            0,
            0,
        )
        .unwrap(),
    };
    assert!(matches!(
        record_resource_graph_usage(
            &primary,
            vec![phase],
            Digest::sha256("workload"),
            Digest::sha256("result"),
            vec![substituted, primary_attempt],
            1,
        ),
        Err(ResourcePlanningError::Invalid(message))
            if message == "resource usage attempt allocation differs from its plan"
    ));
}

#[test]
fn deserialization_revalidates_derived_plan_fields() {
    let plan = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let mut encoded = serde_json::to_value(plan).unwrap();
    encoded["request"]["memory_bytes"] = serde_json::json!(1);
    assert!(serde_json::from_value::<FeaResourcePlan>(encoded).is_err());
}

fn test_accelerator_device(
    class: &str,
    max_allocation_bytes: u64,
) -> runmat_execution::resource::AcceleratorDevice {
    use runmat_execution::resource::*;
    AcceleratorDevice {
        id: AcceleratorDeviceId::new("device:test-accelerator").unwrap(),
        allocation_domain: AcceleratorAllocationDomainId::new("allocation:test-accelerator")
            .unwrap(),
        class: AcceleratorClass::new(class).unwrap(),
        provider: AcceleratorProvider {
            id: AcceleratorProviderId::new("runmat.test").unwrap(),
            version: AcceleratorProviderVersion::new("1.0.0").unwrap(),
            abi_fingerprint: Digest::sha256(b"test-accelerator-abi"),
        },
        max_allocation_bytes,
        features: std::collections::BTreeSet::from([AcceleratorFeature::Compute]),
        inventory_epoch: 1,
    }
}
