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

fn input(partitioning: PartitioningPolicy) -> FeaResourcePlanningInput {
    FeaResourcePlanningInput {
        estimation_model_revision: RESOURCE_ESTIMATION_MODEL_REVISION,
        topology_digest: Digest::sha256("topology"),
        operator_pattern_digest: Digest::sha256("pattern"),
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
            admission: SolverAdmissionResourcePolicy {
                dimension: Count(4),
                factor_nonzero_upper_bound: Count(10),
                update_count_upper_bound: Count(12),
                resident_bytes_upper_bound: ByteCount(200),
            },
        },
        recovery: RecoveryResourcePolicy {
            resident_value_count: Count(4),
            resident_bytes_upper_bound: ByteCount(32),
            scratch_bytes: ByteCount(40),
            retained_bytes: ByteCount(32),
        },
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
            checkpoint_bytes: ByteCount(50),
            artifact_overhead_bytes: ByteCount(20),
            relay_bytes: ByteCount(12),
            result_egress_bytes: ByteCount(30),
            checkpoint_egress_bytes: ByteCount(40),
        },
        worker: WorkerResourcePolicy {
            cpu_millicores: 1_000,
            maximum_wall_milliseconds: 1_000,
            fixed_memory_bytes: ByteCount(100),
            fixed_scratch_bytes: ByteCount(5),
            required_capabilities: BTreeSet::from([Capability::NetworkDenied]),
            accelerators: Vec::new(),
        },
        partitioning,
        accuracy_demand: AccuracyResourceDemand {
            refinement_cycles: Count(0),
            solver_tightenings: Count(0),
            samples: Count(1),
        },
        accuracy_ceiling: AccuracyResourceCeiling {
            maximum_refinement_cycles: 1,
            maximum_solver_tightenings: 1,
            maximum_samples: 4,
            maximum_wall_time_milliseconds: 10_000,
            maximum_resident_bytes: 10_000,
        },
    }
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
        face.provenance.push(PersistentEntityId {
            kind: runmat_meshing_core::PersistentEntityKind::Face,
            source_topology_id: "root".into(),
            assembly_path: vec!["root".into()],
        });
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
    let attached_coordinates = mesh.topology.nodes[0].coordinates_m;
    mesh.topology.nodes[0].provenance.push(PersistentEntityId {
        kind: runmat_meshing_core::PersistentEntityKind::Vertex,
        source_topology_id: "beam_root".into(),
        assembly_path: vec!["root".into()],
    });
    mesh.topology.nodes[0].provenance.sort();
    for face in &mut mesh.topology.boundary_faces {
        face.provenance.push(PersistentEntityId {
            kind: runmat_meshing_core::PersistentEntityKind::Face,
            source_topology_id: "root".into(),
            assembly_path: vec!["root".into()],
        });
        face.provenance.sort();
    }
    mesh.topology.boundary_faces[1]
        .provenance
        .push(PersistentEntityId {
            kind: runmat_meshing_core::PersistentEntityKind::Face,
            source_topology_id: "solid_load".into(),
            assembly_path: vec!["root".into()],
        });
    mesh.topology.boundary_faces[1].provenance.sort();
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.seal_canonical_digest().unwrap();

    let mut model = crate::fixtures::mixed_solid_beam_model(attached_coordinates);
    model.geometry_revision = mesh.geometry.geometry_revision as u32;
    model.connectors = vec![runmat_analysis_core::AnalysisConnector {
        connector_id: "solid_to_beam_spring".into(),
        primary: runmat_analysis_core::RegionTarget::authored("root"),
        secondary: Some(runmat_analysis_core::RegionTarget::authored("span")),
        kind: runmat_analysis_core::ConnectorKind::TranslationalSpring {
            stiffness: runmat_analysis_core::Stiffness::new(2_000.0),
        },
    }];
    let projection = AnalysisProjection::derive(&mesh).unwrap();
    let constraints = CanonicalStructuralConstraintPlan::compile(&model, &mesh).unwrap();
    (model, mesh, projection, constraints)
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
        vec![Count(144), Count(12), Count(12)]
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
            cost: runmat_analysis_core::ObservationCostEstimate {
                operation_count: 1_000,
                peak_resident_bytes: 1_000_000,
                stored_bytes: 1_000_000,
                transfer_bytes: 1_000_000,
                collective_count: 0,
            },
        },
    )];
    let field_definitions = crate::post::fields::structural_observable_field_definitions().unwrap();
    let observations = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &field_definitions,
        ObservationLimits::default(),
    )
    .unwrap();
    let workload = crate::execution::StructuralLinearWorkload {
        schema_version: 1,
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: crate::contracts::ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        algebra_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        initial_guess_transition: None,
    };

    let plan = build_structural_resource_plan(Some(&mesh), &constraints, &observations, &workload)
        .unwrap();
    let mut serial_workload = workload.clone();
    serial_workload.observation_execution =
        crate::results::observation::ObservationExecutionPolicy::serial();
    let serial_plan =
        build_structural_resource_plan(Some(&mesh), &constraints, &observations, &serial_workload)
            .unwrap();

    assert_eq!(
        plan.input.topology_digest,
        constraints.topology.discretization_digest()
    );
    assert_eq!(plan.input.sparse.rows, Count(12));
    assert_eq!(
        plan.input.sparse.operator_nonzeros,
        vec![Count(144), Count(12), Count(12)]
    );
    assert_eq!(
        plan.input.topology,
        TopologyResourceFacts::from_projection(&projection).unwrap()
    );
    assert_eq!(
        plan.input.constraints,
        ConstraintResourceFacts::from_plan(&constraints).unwrap()
    );
    assert!(plan.input.observations.transfer_bytes > serial_plan.input.observations.transfer_bytes);
    assert_ne!(plan.identity, serial_plan.identity);
    let operators =
        StructuralSymbolicOperators::derive(&projection, &constraints, &workload.model, &mesh)
            .unwrap();
    let expected_admission =
        crate::algebra::CoercivitySymbolicPlan::derive(operators.stiffness_upper_bound()).unwrap();
    let SolveResourcePolicy::Iterative { admission, .. } = &plan.input.solve else {
        panic!("structural preflight must authorize iterative solver admission");
    };
    assert_eq!(admission.dimension, Count(expected_admission.dimension()));
    assert_eq!(
        admission.factor_nonzero_upper_bound,
        Count(expected_admission.factor_nonzero_count())
    );
    assert_eq!(
        admission.update_count_upper_bound,
        Count(expected_admission.update_count())
    );
    assert_eq!(
        admission.resident_bytes_upper_bound,
        ByteCount(expected_admission.resident_bytes_upper_bound())
    );
    plan.validate_identity().unwrap();
}

#[test]
fn mixed_preflight_binds_complete_topology_and_connector_sparsity() {
    let (model, mesh, projection, constraints) = mixed_structural_fixture();
    let observations = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &[],
        ObservationLimits::default(),
    )
    .unwrap();
    let workload = crate::execution::StructuralLinearWorkload {
        schema_version: 1,
        step_id: model.sequence.steps[0].step_id.clone(),
        model: model.clone(),
        backend: crate::contracts::ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        algebra_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        initial_guess_transition: None,
    };
    let plan = build_structural_resource_plan(Some(&mesh), &constraints, &observations, &workload)
        .unwrap();

    assert_eq!(plan.input.topology.nodes, Count(5));
    assert_eq!(plan.input.topology.elements, Count(2));
    assert_eq!(plan.input.sparse.rows, Count(21));
    assert_eq!(
        plan.input.topology_digest,
        constraints.topology.discretization_digest()
    );
    assert_eq!(
        plan.identity,
        build_structural_resource_plan(Some(&mesh), &constraints, &observations, &workload,)
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
fn explicit_preflight_derives_complete_bounds_without_a_solver_mesh() {
    let model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    let constraints = CanonicalStructuralConstraintPlan::compile_for_model(&model, None).unwrap();
    let observations = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        None,
        &[],
        ObservationLimits::default(),
    )
    .unwrap();
    let workload = crate::execution::StructuralLinearWorkload {
        schema_version: 1,
        step_id: model.sequence.steps[0].step_id.clone(),
        model: model.clone(),
        backend: crate::contracts::ComputeBackend::Cpu,
        formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
        preconditioner: crate::solve::preconditioner::SpdPreconditionerKind::Jacobi,
        algebra_backend: crate::solve::backend::kind::LinearAlgebraBackendKind::CpuReference,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        require_solver_mesh_for_solid: true,
        initial_guess_transition: None,
    };
    let plan =
        build_structural_resource_plan(None, &constraints, &observations, &workload).unwrap();
    let structural = model.structural.as_ref().unwrap();

    assert_eq!(
        plan.input.topology_digest,
        constraints.topology.discretization_digest()
    );
    assert_eq!(
        plan.input.topology.nodes,
        Count(structural.nodes.len() as u64)
    );
    assert_eq!(
        plan.input.topology.elements,
        Count(structural.elements.len() as u64)
    );
    assert_eq!(plan.input.topology.boundary_faces, Count(0));
    assert_eq!(
        plan.input.sparse.rows,
        Count((structural.nodes.len() * 6) as u64)
    );
    assert_eq!(
        plan.identity,
        build_structural_resource_plan(None, &constraints, &observations, &workload)
            .unwrap()
            .identity
    );
}

#[test]
fn exact_small_estimate_separates_all_resource_classes() {
    let plan = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();

    assert_eq!(plan.breakdown.topology_bytes, ByteCount(1_088));
    assert_eq!(plan.breakdown.csr_row_offset_bytes, ByteCount(120));
    assert_eq!(plan.breakdown.csr_column_index_bytes, ByteCount(144));
    assert_eq!(plan.breakdown.csr_value_bytes, ByteCount(144));
    assert_eq!(plan.breakdown.assembly_contribution_bytes, ByteCount(240));
    assert_eq!(plan.breakdown.solve_vector_bytes, ByteCount(160));
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
    assert_eq!(plan.request.memory_bytes, 2_108);
    assert_eq!(plan.request.scratch_bytes, 45);
    assert_eq!(plan.request.max_artifact_bytes, 302);
    assert_eq!(plan.request.max_egress_bytes, 220);
    assert_eq!(plan.request.max_relay_bytes, 12);
    plan.validate_identity().unwrap();
}

#[test]
fn overflow_and_impossible_sparse_facts_are_rejected() {
    let mut overflow = input(PartitioningPolicy::Monolithic);
    overflow.sparse.rows = Count(u64::MAX);
    overflow.sparse.columns = Count(u64::MAX);
    assert!(matches!(
        build_resource_plan(overflow),
        Err(ResourcePlanningError::Overflow("sparse dense bound"))
    ));

    let mut impossible = input(PartitioningPolicy::Monolithic);
    impossible.sparse.operator_nonzeros[0] = Count(17);
    assert!(matches!(
        build_resource_plan(impossible),
        Err(ResourcePlanningError::Invalid(_))
    ));
}

#[test]
fn understated_resident_bounds_are_rejected() {
    let mut topology = input(PartitioningPolicy::Monolithic);
    topology.topology.estimated_resident_bytes = ByteCount(1);
    assert!(matches!(
        build_resource_plan(topology),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut constraints = input(PartitioningPolicy::Monolithic);
    constraints.constraints.estimated_resident_bytes = ByteCount(1);
    assert!(matches!(
        build_resource_plan(constraints),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut contributions = input(PartitioningPolicy::Monolithic);
    contributions.assembly.contribution_bytes_upper_bound = ByteCount(1);
    assert!(matches!(
        build_resource_plan(contributions),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut recovery = input(PartitioningPolicy::Monolithic);
    recovery.recovery.resident_bytes_upper_bound = ByteCount(1);
    assert!(matches!(
        build_resource_plan(recovery),
        Err(ResourcePlanningError::Invalid(_))
    ));

    let mut preconditioner = input(PartitioningPolicy::Monolithic);
    preconditioner.solve = SolveResourcePolicy::Iterative {
        vector_count: Count(2),
        scalar: NumericalScalar::RealF64,
        preconditioner: PreconditionerStorage::SparseCsr {
            nonzero_upper_bound: Count(10),
            resident_bytes_upper_bound: ByteCount(1),
        },
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
    factor.solve = SolveResourcePolicy::DirectFactorization {
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
    } = &mut admission.solve
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
    let real = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let mut complex_input = input(PartitioningPolicy::Monolithic);
    complex_input.solve = SolveResourcePolicy::Iterative {
        vector_count: Count(5),
        scalar: NumericalScalar::ComplexF64,
        preconditioner: PreconditionerStorage::Diagonal,
        admission: SolverAdmissionResourcePolicy {
            dimension: Count(4),
            factor_nonzero_upper_bound: Count(10),
            update_count_upper_bound: Count(12),
            resident_bytes_upper_bound: ByteCount(280),
        },
    };
    let complex = build_resource_plan(complex_input).unwrap();

    assert_eq!(complex.breakdown.solve_vector_bytes, ByteCount(320));
    assert_eq!(complex.breakdown.preconditioner_bytes, ByteCount(64));
    assert!(complex.request.memory_bytes > real.request.memory_bytes);
}

#[test]
fn accuracy_ceiling_is_enforced_before_inventory_admission() {
    let mut memory = input(PartitioningPolicy::Monolithic);
    memory.accuracy_ceiling.maximum_resident_bytes = 1_000;
    assert!(matches!(
        admit_resource_plan(memory, &inventory(10_000, 10_000)),
        ResourceAdmission::Reject {
            reason: ResourcePlanningError::AccuracyCeiling("resident memory")
        }
    ));

    let mut samples = input(PartitioningPolicy::Monolithic);
    samples.accuracy_demand.samples = Count(5);
    assert!(matches!(
        build_resource_plan(samples),
        Err(ResourcePlanningError::AccuracyCeiling("samples"))
    ));
}

#[test]
fn inventory_admits_or_rejects_a_monolithic_plan() {
    assert!(matches!(
        admit_resource_plan(
            input(PartitioningPolicy::Monolithic),
            &inventory(2_200, 100)
        ),
        ResourceAdmission::Admit { .. }
    ));
    assert!(matches!(
        admit_resource_plan(
            input(PartitioningPolicy::Monolithic),
            &inventory(1_000, 100)
        ),
        ResourceAdmission::Reject {
            reason: ResourcePlanningError::Inventory("memory or scratch")
        }
    ));
}

#[test]
fn operator_domain_work_gets_a_conservative_partition_recommendation() {
    let decision = admit_resource_plan(
        input(PartitioningPolicy::OperatorDomain {
            maximum_partitions: 4,
            replicated_memory_bytes_per_partition: ByteCount(20),
            halo_memory_bytes_per_partition: ByteCount(10),
            replicated_scratch_bytes_per_partition: ByteCount(2),
            halo_scratch_bytes_per_partition: ByteCount(3),
        }),
        &inventory(1_040, 100),
    );
    let ResourceAdmission::RequiresPartitioning {
        minimum_partition_count,
        per_partition_request,
        ..
    } = decision
    else {
        panic!("operator/domain work should receive a partition recommendation");
    };
    assert_eq!(minimum_partition_count, 3);
    assert_eq!(per_partition_request.memory_bytes, 800);
    assert_eq!(per_partition_request.scratch_bytes, 24);
}

#[test]
fn missing_capability_or_accelerator_rejects_before_memory_admission() {
    let no_capability = ResourceInventory {
        capabilities: BTreeSet::new(),
        ..inventory(10_000, 10_000)
    };
    assert!(matches!(
        admit_resource_plan(input(PartitioningPolicy::Monolithic), &no_capability),
        ResourceAdmission::Reject {
            reason: ResourcePlanningError::Capability(Capability::NetworkDenied)
        }
    ));

    let mut accelerated = input(PartitioningPolicy::Monolithic);
    accelerated.worker.accelerators = vec![AcceleratorRequirement {
        class: "gpu".into(),
        count: 1,
        memory_bytes_each: ByteCount(4_096),
    }];
    let inadequate = ResourceInventory {
        accelerators: vec![AcceleratorRequest {
            class: "gpu".into(),
            count: 1,
            memory_bytes_each: 2_048,
        }],
        ..inventory(10_000, 10_000)
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
    changed_input.observations.peak_resident_bytes = ByteCount(400);
    changed_input.observations.stored_bytes = ByteCount(500);
    changed_input.observations.transfer_bytes = ByteCount(450);
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
            egress_bytes: 221,
            relay_bytes: 13,
            retained_byte_millis: 302_001,
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
        class: "gpu".into(),
        count: 1,
        memory_bytes_each: ByteCount(4_096),
    }];
    let plan = build_resource_plan(accelerated).unwrap();
    let actual = AnalysisResourceUsage {
        scheduler: ResourceUsage {
            accelerator_millis: vec![runmat_execution::resource::AcceleratorUsage {
                class: "gpu".into(),
                device_millis: 1_001,
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
fn resource_usage_evidence_binds_plan_workload_result_attempt_and_worker() {
    let plan = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let allocation = plan.request.clone();
    let attempt_id = runmat_execution::identity::AttemptId::derive(&[b"attempt"]);
    let worker_id = runmat_execution::identity::WorkerId::derive(&[b"worker"]);
    let attempt = ResourceUsageAttempt {
        attempt_id,
        worker_id,
        resources: runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            allocation.clone(),
            1,
            0,
            0,
            0,
        )
        .unwrap(),
    };
    let mut partition_allocation = allocation.clone();
    partition_allocation.cpu_millicores = (partition_allocation.cpu_millicores / 2).max(1);
    partition_allocation.memory_bytes = (partition_allocation.memory_bytes / 2).max(1);
    partition_allocation.scratch_bytes /= 2;
    partition_allocation.max_wall_millis = (partition_allocation.max_wall_millis / 2).max(1);
    partition_allocation.max_artifact_bytes /= 2;
    partition_allocation.max_egress_bytes /= 2;
    partition_allocation.max_relay_bytes /= 2;
    let partition_attempt = ResourceUsageAttempt {
        attempt_id: runmat_execution::identity::AttemptId::derive(&[b"partition-attempt"]),
        worker_id: runmat_execution::identity::WorkerId::derive(&[b"partition-worker"]),
        resources: runmat_execution::resource::ResourceAccounting::for_reserved_interval(
            partition_allocation,
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
        vec![partition_attempt, attempt],
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
        Err(ResourcePlanningError::Identity)
    ));
}

#[test]
fn deserialization_revalidates_derived_plan_fields() {
    let plan = build_resource_plan(input(PartitioningPolicy::Monolithic)).unwrap();
    let mut encoded = serde_json::to_value(plan).unwrap();
    encoded["request"]["memory_bytes"] = serde_json::json!(1);
    assert!(serde_json::from_value::<FeaResourcePlan>(encoded).is_err());
}
