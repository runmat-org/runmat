use runmat_analysis_core::{
    AnalysisSequence, CoordinateFrame, CrossPowerSpectralDensity, FormulationRequirement,
    ProcedureStep, RandomExcitationChannel, RandomExcitationChannelId, RandomExcitationId,
    RandomExcitationQuantity, RandomExcitationSpectrum, RandomSpectrumEvidence,
    RandomSpectrumExtrapolation, RandomSpectrumInterpolation, ReferenceStatePolicy,
    SequenceDependencyReason, SpectralDensityConvention, SpectrumDirection, StateTransition,
    StateTransitionMethod, StateUse, TopologyTransitionPolicy, TransitionId,
};
use runmat_execution::Digest;

use crate::{
    execution::{
        AnalysisWorkloadIdentity, CanonicalSemanticState, CanonicalSemanticStateInput,
        CanonicalSemanticStatePayload, CanonicalStateTransition, SequenceIdentity,
        SemanticStatePublicationRequest, SemanticStatePublicationSource, SolverResultIdentity,
        StepIdentity, StructuralModalWorkload, StructuralModalWorkloadInput,
        StructuralRandomResponseAssemblyRequest, StructuralRandomResponseIntervalRequest,
        StructuralRandomResponseIntervalResult, StructuralRandomResponseNumericalReceipt,
        StructuralRandomResponseScope, StructuralRandomResponseSolverResult,
        StructuralRandomResponseWorkload, StructuralRandomResponseWorkloadInput,
    },
    fixtures::{fixture_model, FixtureId},
    pipeline::{
        modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
        structural_random_response::{
            execute_structural_random_response_intervals, prepare_structural_random_response,
        },
    },
    results::observation::ObservationExecutionPolicy,
    solve::dynamics::RandomResponseIntegrationPolicy,
    ComputeBackend,
};

#[test]
fn portable_interval_tasks_are_grouping_invariant_and_bind_receipt() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    let mut modal = ProcedureStep::structural_modal("modes");
    modal.independent_domain =
        runmat_analysis_core::IndependentDomain::Eigenvalue { requested_count: 3 };
    let excitation = random_excitation();
    let response = ProcedureStep::structural_random_response(
        "random",
        excitation.excitation_id.clone(),
        "modes",
        "modal_basis",
    );
    let mut sequence =
        AnalysisSequence::from_steps("modal-random", vec![modal.clone(), response.clone()]);
    sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    sequence.transitions.push(StateTransition {
        transition_id: TransitionId("modes-to-random".into()),
        producer_step: modal.step_id.clone(),
        consumer_step: response.step_id.clone(),
        state_id: runmat_analysis_core::StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    model.sequence = sequence;
    model.random_excitations = vec![excitation];
    let mesh = super::single_tetrahedron_solver_mesh_for_model(&model);
    let topology_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let constraints = crate::constraints::CanonicalStructuralConstraintPlan::compile_for_model(
        &model,
        Some(&mesh),
    )
    .unwrap();

    let modal_workload = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: modal.step_id.clone(),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(3),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let modal_result = crate::execute_structural_modal_workload(&modal_workload, &mesh).unwrap();
    let sequence_identity = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence_identity, &modal).unwrap();
    let source_result =
        SolverResultIdentity::from_step_result(producer, modal_result.basis_identity().digest())
            .unwrap();
    let basis_state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: modal.state_outputs[0].clone(),
        topology_digest,
        source_result,
        payload: CanonicalSemanticStatePayload::ModalBasis {
            basis: Box::new(modal_result.basis().clone()),
        },
    })
    .unwrap();
    let transition = CanonicalStateTransition::bind_upstream(
        &model,
        &response.step_id,
        &basis_state,
        StateUse::ModalBasis,
        topology_digest,
    )
    .unwrap();

    let one = execute_with_group_size(
        &model,
        &response,
        &mesh,
        &constraints,
        &basis_state,
        transition.clone(),
        1,
    );
    let grouped = execute_with_group_size(
        &model,
        &response,
        &mesh,
        &constraints,
        &basis_state,
        transition,
        2,
    );
    assert_eq!(one.numerical(), grouped.numerical());

    let generic_workload = AnalysisWorkloadIdentity::for_test("modal-random").unwrap();
    let receipt =
        StructuralRandomResponseNumericalReceipt::new(generic_workload, &grouped).unwrap();
    receipt
        .validate_against(receipt.source_result(), &grouped)
        .unwrap();
    assert_eq!(
        StructuralRandomResponseNumericalReceipt::from_canonical_bytes(
            &receipt.canonical_bytes().unwrap(),
        )
        .unwrap(),
        receipt
    );
    let consumer = StepIdentity::from_solver_contract(sequence_identity, &response).unwrap();
    let publication = SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::StructuralRandomStatistics,
        consumer,
        response.state_outputs[0].clone(),
        topology_digest,
        receipt.source_result(),
    )
    .unwrap();
    let state = CanonicalSemanticState::from_structural_random_statistics_publication(
        &publication,
        &grouped,
    )
    .unwrap();
    assert!(state.payload.structural_random_statistics().is_some());
    assert_eq!(
        CanonicalSemanticState::from_canonical_bytes(&state.canonical_bytes().unwrap()).unwrap(),
        state
    );
}

fn execute_with_group_size(
    model: &runmat_analysis_core::AnalysisModel,
    response: &ProcedureStep,
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    constraints: &crate::constraints::CanonicalStructuralConstraintPlan,
    basis_state: &CanonicalSemanticState,
    transition: CanonicalStateTransition,
    maximum_intervals_per_task: usize,
) -> StructuralRandomResponseSolverResult {
    let workload = StructuralRandomResponseWorkload::new(StructuralRandomResponseWorkloadInput {
        model: model.clone(),
        step_id: response.step_id.clone(),
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
        observation_execution: ObservationExecutionPolicy::scalable(),
        modal_basis_transition: transition,
        integration_policy: integration_policy(),
        maximum_intervals_per_task,
    })
    .unwrap();
    assert_eq!(
        StructuralRandomResponseWorkload::from_canonical_bytes(
            &workload.canonical_bytes().unwrap(),
        )
        .unwrap(),
        workload
    );
    let prepared =
        prepare_structural_random_response(&workload, mesh, basis_state, constraints).unwrap();
    let scope = prepared.scope().clone();
    assert_eq!(
        StructuralRandomResponseScope::from_canonical_bytes(&scope.canonical_bytes().unwrap())
            .unwrap(),
        scope
    );
    let mut tasks = (0..scope.task_ranges().len())
        .map(|ordinal| {
            let request =
                StructuralRandomResponseIntervalRequest::new(&workload, scope.clone(), ordinal)
                    .unwrap();
            assert_eq!(
                StructuralRandomResponseIntervalRequest::from_canonical_bytes(
                    &request.canonical_bytes().unwrap(),
                )
                .unwrap(),
                request
            );
            let task = execute_structural_random_response_intervals(&workload, &prepared, &request)
                .unwrap();
            assert_eq!(
                StructuralRandomResponseIntervalResult::from_canonical_bytes(
                    &task.canonical_bytes().unwrap(),
                )
                .unwrap(),
                task
            );
            task
        })
        .collect::<Vec<_>>();
    let missing = tasks[..tasks.len() - 1].to_vec();
    assert!(StructuralRandomResponseAssemblyRequest::new(scope.clone(), missing).is_err());
    tasks.reverse();
    let assembly = StructuralRandomResponseAssemblyRequest::new(scope, tasks).unwrap();
    assert_eq!(assembly.tasks()[0].request().task_ordinal(), 0);
    assert_eq!(
        StructuralRandomResponseAssemblyRequest::from_canonical_bytes(
            &assembly.canonical_bytes().unwrap(),
        )
        .unwrap(),
        assembly
    );
    let result = StructuralRandomResponseSolverResult::new(&workload, assembly).unwrap();
    assert_eq!(
        StructuralRandomResponseSolverResult::from_canonical_bytes(
            &result.canonical_bytes().unwrap(),
        )
        .unwrap(),
        result
    );
    result
}

fn integration_policy() -> RandomResponseIntegrationPolicy {
    RandomResponseIntegrationPolicy {
        modal_damping_ratio: 0.04,
        minimum_cumulative_mass_ratio: 0.0,
        relative_variance_tolerance: 1.0e-5,
        absolute_variance_tolerance: 1.0e-14,
        maximum_refinement_levels: 14,
    }
}

fn random_excitation() -> RandomExcitationSpectrum {
    let channel = RandomExcitationChannel {
        channel_id: RandomExcitationChannelId("base-x".into()),
        quantity: RandomExcitationQuantity::BaseAcceleration,
        direction: SpectrumDirection {
            frame: CoordinateFrame::Global,
            vector: [1.0, 0.0, 0.0],
        },
    };
    RandomExcitationSpectrum {
        excitation_id: RandomExcitationId("broadband-base".into()),
        frequencies_hz: vec![1.0e-3, 100.0, 1_000.0, 10_000.0],
        channels: vec![channel],
        spectral_density: vec![CrossPowerSpectralDensity::diagonal(&[2.5e-4]); 4],
        convention: SpectralDensityConvention::OneSidedPerHertz,
        interpolation: RandomSpectrumInterpolation::LinearFrequency,
        extrapolation: RandomSpectrumExtrapolation::Reject,
        evidence: RandomSpectrumEvidence {
            max_absolute_density_error: Some(0.0),
            equivalent_degrees_of_freedom: None,
        },
        provenance_digest: "sha256:test-broadband-base".into(),
    }
}
