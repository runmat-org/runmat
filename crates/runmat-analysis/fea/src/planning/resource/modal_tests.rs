use runmat_analysis_core::{
    FormulationRequirement, IndependentDomain, ProcedureStep, ProcedureStepId,
};
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
    FeaResourcePhase, PartitioningPolicy, PreparedInputResourceFacts, PreparedOutputResourceFacts,
    PreparedOutputSetResourceFacts, PreparedStructuralModalResourceFacts, SolveResourcePolicy,
};

fn prepared(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    constraints: &CanonicalStructuralConstraintPlan,
    workload: &StructuralModalWorkload,
) -> PreparedStructuralModalResourceFacts {
    let result =
        structural_modal_result_payload_bytes_upper_bound(Some(mesh), constraints, workload)
            .unwrap();
    let receipt = structural_modal_receipt_payload_bytes_upper_bound();
    PreparedStructuralModalResourceFacts {
        inputs: PreparedInputResourceFacts {
            encoded_payload_bytes: ByteCount(1024),
            transport_closure_bytes: ByteCount(2048),
            decoded_resident_bytes_admission_budget: ByteCount(128 * 1024),
        },
        result_output: PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(result),
            transport_closure_bytes_upper_bound: ByteCount(result + 128 * 1024),
            construction_resident_bytes_admission_budget: ByteCount(result * 2),
        },
        receipt_output: PreparedOutputResourceFacts {
            encoded_payload_bytes_upper_bound: ByteCount(receipt),
            transport_closure_bytes_upper_bound: ByteCount(receipt + 128 * 1024),
            construction_resident_bytes_admission_budget: ByteCount(receipt * 2),
        },
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
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let truncation = ModalTruncationPolicy::requested(requested_modes);
    let mut step = ProcedureStep::structural_modal("modes");
    step.independent_domain = IndependentDomain::Eigenvalue {
        requested_count: requested_modes,
    };
    step.observations.clear();
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
        formulation_requirement: FormulationRequirement::Experimental,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation,
        convergence: ModalConvergencePolicy::exploratory(),
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
        &[],
        understated_receipt,
    )
    .is_err());
    let mut substituted_observations = observations.clone();
    substituted_observations.step_id = ProcedureStepId("another-step".into());
    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &substituted_observations,
        &small_workload,
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
fn modal_accelerator_admission_rejects_missing_short_and_cpu_claims() {
    let (cpu_workload, mesh, constraints, observations) = modal_case(2, ComputeBackend::Cpu);
    let spurious = AcceleratorRequest {
        class: "test-gpu".into(),
        count: 1,
        memory_bytes_each: u64::MAX,
    };
    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &cpu_workload,
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
        class: "test-gpu".into(),
        count: 1,
        memory_bytes_each: required - 1,
    };
    assert!(build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &gpu_workload,
        std::slice::from_ref(&short),
        prepared(&mesh, &constraints, &gpu_workload),
    )
    .is_err());
    let exact = AcceleratorRequest {
        memory_bytes_each: required,
        ..short
    };
    let accelerated = build_structural_modal_resource_plan(
        Some(&mesh),
        &constraints,
        &observations,
        &gpu_workload,
        std::slice::from_ref(&exact),
        prepared(&mesh, &constraints, &gpu_workload),
    )
    .unwrap();
    assert_eq!(accelerated.request.accelerators, vec![exact]);
}
