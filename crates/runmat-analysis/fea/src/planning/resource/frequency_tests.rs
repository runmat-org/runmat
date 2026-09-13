use runmat_analysis_core::{
    FormulationRequirement, IndependentDomain, ProcedureStep, ProcedureStepId,
    ReferenceStatePolicy, SequenceDependency, SequenceDependencyReason, StateId, StateTransition,
    StateTransitionMethod, TopologyTransitionPolicy, TransitionId,
};
use runmat_execution::resource::AcceleratorRequest;
use runmat_execution::Digest;

use crate::{
    constraints::CanonicalStructuralConstraintPlan,
    execution::{
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        SequenceIdentity, SolverResultIdentity, StepIdentity, StructuralFrequencyEntryRequest,
        StructuralFrequencySolverPlan, StructuralFrequencyWorkload,
        StructuralFrequencyWorkloadInput,
    },
    fixtures::{fixture_model, FixtureId},
    frequency::FrequencyResponsePlan,
    pipeline::structural_frequency::prepare_structural_frequency,
    results::observation::ObservationExecutionPolicy,
    solve::dynamics::DirectHarmonicPolicy,
    ComputeBackend, ModalSolveOptions, ModalTruncationPolicy,
};

use super::{
    build_structural_frequency_entry_resource_plan, FeaResourcePhase, SolveResourcePolicy,
};

fn direct_case(
    restart: usize,
) -> (
    StructuralFrequencyWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalStructuralConstraintPlan,
    StructuralFrequencyEntryRequest,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_harmonic("frequency");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0]);
    model.sequence.steps = vec![step];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: runmat_analysis_core::ProcedureStepId("frequency".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::Direct {
            policy: DirectHarmonicPolicy {
                restart,
                ..DirectHarmonicPolicy::default()
            },
        },
    })
    .unwrap();
    let prepared = prepare_structural_frequency(&workload, &mesh).unwrap();
    let entry = workload.plan().entries().unwrap()[0];
    let request = StructuralFrequencyEntryRequest::new(
        &workload,
        prepared.study,
        prepared.step,
        prepared.execution_plan,
        prepared.scope,
        entry,
    )
    .unwrap();
    (workload, mesh, constraints, request)
}

fn modal_case() -> (
    StructuralFrequencyWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalStructuralConstraintPlan,
    StructuralFrequencyEntryRequest,
    CanonicalSemanticState,
) {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let modal_step = ProcedureStep::structural_modal("modes");
    let mut frequency_step =
        ProcedureStep::structural_modal_harmonic("frequency", "modes", "modal_basis");
    frequency_step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0]);
    model.sequence.steps = vec![modal_step, frequency_step];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("modes".into()),
        successor: ProcedureStepId("frequency".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-to-frequency".into()),
        producer_step: ProcedureStepId("modes".into()),
        consumer_step: ProcedureStepId("frequency".into()),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    let mesh = crate::tests::single_tetrahedron_solver_mesh_for_model(&model);
    let constraints =
        CanonicalStructuralConstraintPlan::compile_for_model(&model, Some(&mesh)).unwrap();
    let modal = crate::run_modal_with_options(
        &model,
        ComputeBackend::Cpu,
        ModalSolveOptions {
            truncation: ModalTruncationPolicy::requested(3),
            solver_mesh: Some(mesh.clone()),
            formulation_requirement: FormulationRequirement::Experimental,
            ..ModalSolveOptions::default()
        },
    )
    .unwrap();
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, modal.basis.identity().digest()).unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: model.sequence.steps[0].state_outputs[0].clone(),
        topology_digest: Digest::from_bytes(*mesh.canonical_digest.bytes()),
        source_result,
        payload: CanonicalSemanticStatePayload::ModalBasis {
            basis: Box::new(modal.basis),
        },
    })
    .unwrap();
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: ProcedureStepId("frequency".into()),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::ModalReduced {
            basis_state: state.identity,
        },
    })
    .unwrap();
    let prepared = prepare_structural_frequency(&workload, &mesh).unwrap();
    let request = StructuralFrequencyEntryRequest::new(
        &workload,
        prepared.study,
        prepared.step,
        prepared.execution_plan,
        prepared.scope,
        workload.plan().entries().unwrap()[0],
    )
    .unwrap();
    (workload, mesh, constraints, request, state)
}

#[test]
fn direct_frequency_entry_plan_binds_request_and_concrete_gmres_growth() {
    let (small_workload, mesh, constraints, small_request) = direct_case(2);
    let small = build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &small_workload,
        &small_request,
        None,
        &[],
    )
    .unwrap();
    assert_eq!(
        small.input.phase,
        FeaResourcePhase::StructuralFrequencyEntry(small_request.identity())
    );
    let SolveResourcePolicy::RestartedGmres {
        block_dimension,
        restart,
        workspace_bytes_upper_bound,
        ..
    } = small.input.solve
    else {
        panic!("direct harmonic work must publish its GMRES workspace")
    };
    assert_eq!(restart.0, 2);
    assert!(block_dimension.0 > 0);
    assert!(workspace_bytes_upper_bound.0 > 0);
    assert!(small.request.accelerators.is_empty());

    let (large_workload, large_mesh, large_constraints, large_request) = direct_case(4);
    let large = build_structural_frequency_entry_resource_plan(
        &large_mesh,
        &large_constraints,
        &large_workload,
        &large_request,
        None,
        &[],
    )
    .unwrap();
    assert!(
        large.breakdown.solve_workspace_bytes.0 > small.breakdown.solve_workspace_bytes.0,
        "larger Krylov bases must increase the admitted peak"
    );
}

#[test]
fn frequency_entry_preflight_rejects_accelerator_and_request_substitution() {
    let (workload, mesh, constraints, request) = direct_case(2);
    assert!(build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &request,
        None,
        &[AcceleratorRequest {
            class: "gpu".into(),
            count: 1,
            memory_bytes_each: 1024,
        }],
    )
    .is_err());

    let (other_workload, _, _, other_request) = direct_case(3);
    assert!(build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &other_workload,
        &request,
        None,
        &[],
    )
    .is_err());
    assert_ne!(request.identity(), other_request.identity());
}

#[test]
fn modal_frequency_entry_plan_accounts_for_the_exact_basis_and_dense_solver() {
    let (workload, mesh, constraints, request, state) = modal_case();
    let plan = build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &request,
        Some(&state),
        &[],
    )
    .unwrap();
    let mode_count = u64::try_from(state.payload.modal_basis().unwrap().modes().len()).unwrap();
    let SolveResourcePolicy::ModalDenseHarmonic {
        full_dimension,
        mode_count: planned_modes,
        workspace_bytes_upper_bound,
    } = plan.input.solve
    else {
        panic!("modal harmonic work must publish its dense workspace")
    };
    assert_eq!(
        full_dimension.0,
        u64::try_from(request.scope().physical_dof_count()).unwrap()
    );
    assert_eq!(planned_modes.0, mode_count);
    assert!(workspace_bytes_upper_bound.0 > 0);
    assert!(build_structural_frequency_entry_resource_plan(
        &mesh,
        &constraints,
        &workload,
        &request,
        None,
        &[],
    )
    .is_err());
}
