mod allocation;
mod capacity;
mod compiler;
mod local_energy;
mod proof_workspace;
mod solve_path;
use runmat_analysis_core::{IndependentDomain, ProcedureStep, ProcedureStepId};
use runmat_execution::resource::AcceleratorRequest;

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        AnalysisNumericalObjectKind, StructuralModalWorkload, StructuralModalWorkloadInput,
    },
    fixtures::{fixture_model, FixtureId},
    pipeline::modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
    post::fields::structural_observable_field_definitions,
    results::observation::{
        resolve_structural_observation_plan, ObservationExecutionPolicy, ObservationLimits,
    },
    ComputeBackend,
};

use super::{
    build_structural_modal_resource_plan, structural_modal_receipt_payload_bytes_upper_bound,
    structural_modal_result_payload_bytes_upper_bound, ByteCount, FeaResourceDemand,
    FeaResourcePhase, PartitioningPolicy, PreparedInputResourceFacts,
    PreparedOutputSetResourceFacts, PreparedStructuralModalResourceFacts, SolveResourcePolicy,
};

fn prepared(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralModalWorkload,
) -> PreparedStructuralModalResourceFacts {
    let result =
        super::structural_modal_result_output_resource_facts(Some(mesh), constraints, workload)
            .unwrap();
    let receipt = structural_modal_receipt_payload_bytes_upper_bound();
    let profiles = crate::execution::structural_state_publication_profiles_encoding_profile(
        workload.model(),
        workload.procedure_step().unwrap(),
        &[crate::execution::SemanticStatePublicationSource::ModalBasis],
        runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )
    .unwrap()
    .encoded_bytes_upper_bound;
    PreparedStructuralModalResourceFacts {
        local_execution: super::LocalExecutionPlan::serial(),
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(1024),
            transport_closure_bytes: ByteCount(2048),
            decoded_resident_bytes_admission_budget: ByteCount(128 * 1024),
        },
        result_output: result,
        receipt_output: super::projected_canonical_output_resource_facts(receipt).unwrap(),
        state_profiles_output: super::projected_canonical_output_resource_facts(profiles).unwrap(),
        output_set: PreparedOutputSetResourceFacts {
            encoded_bytes_upper_bound: ByteCount(64 * 1024),
        },
    }
}

fn modal_case(
    requested_modes: usize,
    backend: ComputeBackend,
) -> (
    StructuralModalWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalStructuralConstraintPlan,
    crate::results::observation::ResolvedObservationPlan,
) {
    modal_case_with(
        requested_modes,
        backend,
        runmat_analysis_core::ModalSolverMethod::Auto,
    )
}

fn modal_case_with(
    requested_modes: usize,
    backend: ComputeBackend,
    solver_method: runmat_analysis_core::ModalSolverMethod,
) -> (
    StructuralModalWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalStructuralConstraintPlan,
    crate::results::observation::ResolvedObservationPlan,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let truncation = ModalTruncationPolicy::requested(requested_modes);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: requested_modes,
    };
    step.observations.clear();
    if solver_method != runmat_analysis_core::ModalSolverMethod::Auto {
        step.settings = runmat_analysis_core::ProcedureSettings::StructuralModalBasis(
            runmat_analysis_core::StructuralModalBasisSettings {
                residual_vector_directions: Vec::new(),
                solver_method,
            },
        );
    }
    model.sequence.steps = vec![step];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
    let fields = structural_observable_field_definitions().unwrap();
    let observations = resolve_structural_observation_plan(
        &model.sequence.steps[0],
        &model,
        Some(&mesh),
        &constraints,
        &fields,
        ObservationLimits::default(),
    )
    .unwrap();
    let workload = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model,
        step_id: ProcedureStepId("modes".into()),
        backend,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation,
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    (workload, mesh, constraints, observations)
}

#[test]
fn modal_plan_binds_k_m_probe_and_monolithic_peak() {
    let (small_workload, mesh, constraints, observations) = modal_case(1, ComputeBackend::Cpu);
    let small = build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &small_workload,
        None,
        &[],
        prepared(&mesh, &constraints, &small_workload),
    )
    .unwrap();
    assert_eq!(
        small.input.phase,
        FeaResourcePhase::StructuralModalExtraction(small_workload.identity())
    );
    let FeaResourceDemand::StructuralModalExtraction(demand) = &small.input.demand else {
        panic!("modal planning must preserve its typed demand")
    };
    assert_eq!(demand.operator.partitioning, PartitioningPolicy::Monolithic);
    assert_eq!(
        demand.operator.assembly.fixed_scratch_bytes_per_worker.0,
        super::assembly_sizing::FIXED_ASSEMBLY_SCRATCH_BYTES_PER_WORKER
    );
    assert!(demand.operator.sparse.operator_nonzeros.len() >= 2);
    let SolveResourcePolicy::SymmetricSubspaceIteration {
        requested_mode_count,
        initial_subspace_dimension,
        maximum_subspace_dimension,
        boundary_probe_width,
        dense_projected_value_count_upper_bound,
        ..
    } = demand.operator.solve
    else {
        panic!("modal planning must publish its subspace policy")
    };
    assert_eq!(requested_mode_count.0, 1);
    assert_eq!(boundary_probe_width.0, 4);
    assert_eq!(
        dense_projected_value_count_upper_bound.0,
        maximum_subspace_dimension.0 * maximum_subspace_dimension.0
    );
    assert!(initial_subspace_dimension.0 > requested_mode_count.0);
    assert!(small
        .input
        .demand
        .output_facts(AnalysisNumericalObjectKind::SolverResult)
        .is_some());
    assert!(small
        .input
        .demand
        .output_facts(AnalysisNumericalObjectKind::NumericalReceipt)
        .is_some());
    small.validate_identity().unwrap();

    let mut understated = prepared(&mesh, &constraints, &small_workload);
    understated
        .result_output
        .encoded_payload_bytes_upper_bound
        .0 -= 1;
    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &small_workload,
        None,
        &[],
        understated,
    )
    .is_err());
    let mut understated_receipt = prepared(&mesh, &constraints, &small_workload);
    understated_receipt
        .receipt_output
        .encoded_payload_bytes_upper_bound
        .0 -= 1;
    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &small_workload,
        None,
        &[],
        understated_receipt,
    )
    .is_err());
    let mut substituted_observations = observations.clone().into_contents();
    let mut understated_profiles = prepared(&mesh, &constraints, &small_workload);
    understated_profiles
        .state_profiles_output
        .encoded_payload_bytes_upper_bound
        .0 -= 1;
    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &small_workload,
        None,
        &[],
        understated_profiles,
    )
    .is_err());
    substituted_observations.step_id = ProcedureStepId("another-step".into());
    assert!(
        crate::results::observation::ResolvedObservationPlan::try_from(
            substituted_observations.clone()
        )
        .is_err()
    );
    substituted_observations.identity = substituted_observations.derive_identity().unwrap();
    let substituted_observations =
        crate::results::observation::ResolvedObservationPlan::try_from(substituted_observations)
            .unwrap();
    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &substituted_observations,
        &small_workload,
        None,
        &[],
        prepared(&mesh, &constraints, &small_workload),
    )
    .is_err());

    let (large_workload, large_mesh, large_constraints, large_observations) =
        modal_case(3, ComputeBackend::Cpu);
    let large = build_structural_modal_resource_plan(
        Some(&large_mesh),
        &large_constraints,
        &large_observations,
        &large_workload,
        None,
        &[],
        prepared(&large_mesh, &large_constraints, &large_workload),
    )
    .unwrap();
    let SolveResourcePolicy::SymmetricSubspaceIteration {
        initial_subspace_dimension: large_initial,
        ..
    } = large.input.structural_operator().unwrap().solve
    else {
        panic!("modal planning must publish its subspace policy")
    };
    assert!(large_initial.0 >= initial_subspace_dimension.0);
    assert!(large.request.memory_bytes >= small.request.memory_bytes);
}

#[test]
fn modal_plan_accounts_for_requested_residual_vectors() {
    let (base, mesh, constraints, observations) = modal_case(2, ComputeBackend::Cpu);
    let enriched = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: base.model().clone(),
        step_id: base.step_id().clone(),
        backend: base.backend(),
        observation_execution: base.observation_execution(),
        truncation: base.truncation(),
        convergence: base.convergence(),
        residual_vector_directions: vec![runmat_analysis_core::SpectrumDirection {
            frame: runmat_analysis_core::CoordinateFrame::Global,
            vector: [1.0, 0.5, 0.25],
        }],
        prestress_transition: None,
    })
    .unwrap();
    let base_plan = build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &base,
        None,
        &[],
        prepared(&mesh, &constraints, &base),
    )
    .unwrap();
    let enriched_plan = build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &enriched,
        None,
        &[],
        prepared(&mesh, &constraints, &enriched),
    )
    .unwrap();
    let FeaResourceDemand::StructuralModalExtraction(base_demand) = &base_plan.input.demand else {
        panic!("modal plan must retain typed demand")
    };
    let FeaResourceDemand::StructuralModalExtraction(enriched_demand) = &enriched_plan.input.demand
    else {
        panic!("modal plan must retain typed demand")
    };
    assert_eq!(base_demand.residual_vector_count.0, 0);
    assert_eq!(
        base_demand
            .residual_vector_static_iteration_count_upper_bound
            .0,
        0
    );
    assert_eq!(enriched_demand.residual_vector_count.0, 1);
    assert!(
        enriched_demand
            .residual_vector_static_iteration_count_upper_bound
            .0
            >= 1_000
    );
    assert_eq!(
        enriched_demand
            .residual_vector_mass_orthogonalization_passes
            .0,
        2
    );
    assert!(
        enriched_demand.modal_output_value_count_upper_bound.0
            > base_demand.modal_output_value_count_upper_bound.0
    );
    compiler::assert_residual_work_storage(&base_plan, &enriched_plan);
}

#[test]
fn prestressed_modal_plan_accounts_for_exact_geometric_state_and_tangent() {
    let (workload, mesh, state) = crate::tests::prepared_prestressed_modal_case(-1.0);
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(workload.model(), Some(&mesh))
            .unwrap();
    let fields = structural_observable_field_definitions().unwrap();
    let observations = resolve_structural_observation_plan(
        workload.procedure_step().unwrap(),
        workload.model(),
        Some(&mesh),
        &constraints,
        &fields,
        ObservationLimits::default(),
    )
    .unwrap();
    let plan = build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &workload,
        Some(&state),
        &[],
        prepared(&mesh, &constraints, &workload),
    )
    .unwrap();
    let FeaResourceDemand::StructuralModalExtraction(demand) = plan.input.demand else {
        panic!("prestressed modal planning must retain typed modal demand")
    };
    let linearization = state.payload.structural_linearization().unwrap();
    assert_eq!(
        demand.prestress_linearization,
        Some(linearization.identity())
    );
    assert_eq!(
        demand.geometric_stiffness_nonzero_count.0,
        linearization.geometric_stiffness().values().len() as u64
    );
    assert!(demand.reduced_tangent_nonzero_count_upper_bound.0 > 0);

    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &workload,
        None,
        &[],
        prepared(&mesh, &constraints, &workload),
    )
    .is_err());
    let mut substituted = state;
    substituted.topology_digest = runmat_execution::Digest::sha256(b"substituted-topology");
    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &workload,
        Some(&substituted),
        &[],
        prepared(&mesh, &constraints, &workload),
    )
    .is_err());
}

#[test]
fn modal_accelerator_admission_rejects_missing_short_and_cpu_claims() {
    let (cpu_workload, mesh, constraints, observations) = modal_case(2, ComputeBackend::Cpu);
    let spurious = AcceleratorRequest {
        class: runmat_execution::resource::AcceleratorClass::new("test-gpu").unwrap(),
        count: 1,
        minimum_allocation_bytes: u64::MAX,
        provider: None,
        required_features: Default::default(),
    };
    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &cpu_workload,
        None,
        std::slice::from_ref(&spurious),
        prepared(&mesh, &constraints, &cpu_workload),
    )
    .is_err());

    let (gpu_workload, mesh, constraints, observations) = modal_case(2, ComputeBackend::Gpu);
    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &gpu_workload,
        None,
        &[],
        prepared(&mesh, &constraints, &gpu_workload),
    )
    .is_err());

    let cpu_projection = {
        let (workload, mesh, constraints, observations) = modal_case(2, ComputeBackend::Cpu);
        build_structural_modal_resource_plan(
            Some(&mesh),
            &constraints,
            &observations,
            &workload,
            None,
            &[],
            prepared(&mesh, &constraints, &workload),
        )
        .unwrap()
    };
    let SolveResourcePolicy::SymmetricSubspaceIteration {
        reduced_dimension, ..
    } = cpu_projection.input.structural_operator().unwrap().solve
    else {
        panic!("modal planning must publish its subspace policy")
    };
    let facts = super::structural_facts::derive_structural_resource_facts(
        gpu_workload.model(),
        Some(&mesh),
        &constraints,
    )
    .unwrap();
    let reduction = facts.constraint_matrix.affine_reduction().unwrap();
    let reduced_stiffness = reduction
        .reduce_sparse_pattern(facts.symbolic.stiffness_upper_bound())
        .unwrap();
    let stiffness_nonzeros = u64::try_from(reduced_stiffness.column_indices().len()).unwrap();
    let required = super::modal::structural_modal_accelerator_minimum_bytes(
        reduced_dimension.0,
        stiffness_nonzeros,
    )
    .unwrap();
    let short = AcceleratorRequest {
        class: runmat_execution::resource::AcceleratorClass::new("test-gpu").unwrap(),
        count: 1,
        minimum_allocation_bytes: required - 1,
        provider: None,
        required_features: Default::default(),
    };
    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &gpu_workload,
        None,
        std::slice::from_ref(&short),
        prepared(&mesh, &constraints, &gpu_workload),
    )
    .is_err());
    let exact = AcceleratorRequest {
        minimum_allocation_bytes: required,
        provider: None,
        required_features: Default::default(),
        ..short
    };
    let accelerated = build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &gpu_workload,
        None,
        std::slice::from_ref(&exact),
        prepared(&mesh, &constraints, &gpu_workload),
    )
    .unwrap();
    assert_eq!(accelerated.request.accelerators, vec![exact]);
}
