use runmat_analysis_core::{
    AnalysisSequence, CoordinateFrame, CrossPowerSpectralDensity, ProcedureStep,
    RandomExcitationChannel, RandomExcitationChannelId, RandomExcitationId,
    RandomExcitationQuantity, RandomExcitationSpectrum, RandomSpectrumEvidence,
    RandomSpectrumExtrapolation, RandomSpectrumInterpolation, ReferenceStatePolicy,
    SequenceDependencyReason, SpectralDensityConvention, SpectrumDirection, StateTransition,
    StateTransitionMethod, StateUse, TopologyTransitionPolicy, TransitionId,
};
use runmat_execution::Digest;

use crate::{
    execution::{
        AccuracySourceKind, AccuracySourceReceipt, AnalysisWorkloadIdentity,
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        CanonicalStateTransition, SemanticStatePublicationRequest, SemanticStatePublicationSource,
        SequenceIdentity, SolverResultIdentity, StepIdentity, StructuralModalWorkload,
        StructuralModalWorkloadInput, StructuralRandomResponseAssemblyRequest,
        StructuralRandomResponseIntervalRequest, StructuralRandomResponseIntervalResult,
        StructuralRandomResponseNumericalReceipt, StructuralRandomResponseScope,
        StructuralRandomResponseSolverResult, StructuralRandomResponseWorkload,
        StructuralRandomResponseWorkloadInput,
    },
    fixtures::{fixture_model, FixtureId},
    pipeline::{
        modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
        structural_random_response::{
            execute_structural_random_response_intervals, prepare_structural_random_response,
        },
    },
    planning::resource::{
        build_resource_plan, structural_random_response_assembly_payload_bytes_upper_bound,
        structural_random_response_interval_payload_bytes_upper_bound,
        structural_random_response_receipt_payload_bytes_upper_bound, ByteCount, FeaResourceDemand,
        FeaResourcePlan, PreparedInputResourceFacts, PreparedOutputResourceFacts,
        PreparedOutputSetResourceFacts,
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
        runmat_analysis_core::StructuralRandomResponseSettings {
            modal_damping_ratio: 0.05,
            minimum_cumulative_mass_ratio: 0.9,
        },
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

    let basis_publication = SemanticStatePublicationRequest::new(
        SemanticStatePublicationSource::ModalBasis,
        producer,
        modal.state_outputs[0].clone(),
        topology_digest,
        source_result,
    )
    .unwrap();
    let basis_profile = basis_publication
        .encoding_profile(
            crate::execution::SemanticStatePublicationProfileSource::ModalBasis(
                modal_result.basis(),
            ),
        )
        .unwrap();
    assert_eq!(
        basis_profile,
        runmat_canonical_codec::profile_serialized(
            &basis_state,
            runmat_canonical_codec::CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64),
        )
        .unwrap()
    );
    assert!(
        basis_state.canonical_bytes().unwrap().len() as u64
            <= basis_profile.encoded_bytes_upper_bound
    );

    let (one, one_task_workspace) = execute_with_group_size(
        &model,
        &response,
        &mesh,
        &constraints,
        &basis_state,
        transition.clone(),
        1,
    );
    let (grouped, grouped_task_workspace) = execute_with_group_size(
        &model,
        &response,
        &mesh,
        &constraints,
        &basis_state,
        transition,
        2,
    );
    assert_eq!(one.numerical(), grouped.numerical());
    assert_eq!(one_task_workspace, grouped_task_workspace);

    let generic_workload = AnalysisWorkloadIdentity::for_test("modal-random").unwrap();
    let receipt =
        StructuralRandomResponseNumericalReceipt::new(generic_workload, &grouped).unwrap();
    assert!(
        receipt.canonical_bytes().unwrap().len() as u64
            <= structural_random_response_receipt_payload_bytes_upper_bound()
    );
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
    let receipt_bytes = receipt.canonical_bytes().unwrap();
    assert!(matches!(
        AccuracySourceReceipt::from_canonical_bytes(
            AccuracySourceKind::StructuralRandomResponse,
            &receipt_bytes,
        )
        .unwrap(),
        AccuracySourceReceipt::StructuralRandomResponse(_)
    ));
    assert!(AccuracySourceReceipt::from_canonical_bytes(
        AccuracySourceKind::StructuralModalResponseSpectrum,
        &receipt_bytes,
    )
    .is_err());
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
    let profile = publication
        .encoding_profile(
            crate::execution::SemanticStatePublicationProfileSource::StructuralRandomStatistics(
                &grouped,
            ),
        )
        .unwrap();
    assert_eq!(
        profile,
        runmat_canonical_codec::profile_serialized(
            &state,
            runmat_canonical_codec::CanonicalLimits::new(1024 * 1024 * 1024, 4_000_000, 65_536, 64),
        )
        .unwrap()
    );
    assert!(state.canonical_bytes().unwrap().len() as u64 <= profile.encoded_bytes_upper_bound);
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
) -> (StructuralRandomResponseSolverResult, u64) {
    let workload = StructuralRandomResponseWorkload::new(StructuralRandomResponseWorkloadInput {
        model: model.clone(),
        step_id: response.step_id.clone(),
        backend: ComputeBackend::Cpu,
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
    let projection = prepared.resources();
    let scope = prepared.scope().clone();
    assert_eq!(
        StructuralRandomResponseScope::from_canonical_bytes(&scope.canonical_bytes().unwrap())
            .unwrap(),
        scope
    );
    let mut task_workspace_bytes = None;
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
            let payload =
                structural_random_response_interval_payload_bytes_upper_bound(&request).unwrap();
            let plan = projection
                .plan_for_intervals(
                    &request,
                    prepared_inputs(),
                    prepared_output(payload),
                    prepared_output_set(),
                    crate::planning::resource::LocalExecutionPlan::serial(),
                )
                .unwrap();
            let FeaResourceDemand::StructuralRandomResponseInterval(demand) = &plan.input.demand
            else {
                panic!("random-response interval plan has the wrong typed demand");
            };
            assert_eq!(
                demand.response_value_count.0,
                request.scope().plan().value_count() as u64
            );
            assert!(demand.maximum_frequency_evaluation_count.0 >= 3 * demand.interval_count.0);
            assert_eq!(demand.spectral_matrix_complex_value_count.0, 4);
            assert_eq!(
                demand.selected_mode_count.0,
                request.scope().plan().selected_mode_count() as u64
            );
            assert_eq!(
                demand.linear_field_value_count.0 + demand.physical_dof_count.0,
                demand.response_value_count.0
            );
            match task_workspace_bytes {
                Some(expected) => assert_eq!(plan.breakdown.workspace_bytes.0, expected),
                None => task_workspace_bytes = Some(plan.breakdown.workspace_bytes.0),
            }
            assert_eq!(
                FeaResourcePlan::from_canonical_bytes(&plan.canonical_bytes().unwrap()).unwrap(),
                plan
            );
            if ordinal == 0 {
                let mut underreported = plan.input.clone();
                let FeaResourceDemand::StructuralRandomResponseInterval(demand) =
                    &mut underreported.demand
                else {
                    unreachable!()
                };
                demand.response_value_count.0 -= 1;
                assert!(build_resource_plan(underreported).is_err());
                assert!(projection
                    .plan_for_intervals(
                        &request,
                        prepared_inputs(),
                        prepared_output(payload - 1),
                        prepared_output_set(),
                        crate::planning::resource::LocalExecutionPlan::serial(),
                    )
                    .is_err());
            }
            let task = execute_structural_random_response_intervals(&workload, &prepared, &request)
                .unwrap();
            assert!(task.canonical_bytes().unwrap().len() as u64 <= payload);
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
    let result_payload =
        structural_random_response_assembly_payload_bytes_upper_bound(&assembly).unwrap();
    let assembly_plan = projection
        .plan_for_assembly(
            &assembly,
            prepared_inputs(),
            prepared_output(result_payload),
            prepared_output(structural_random_response_receipt_payload_bytes_upper_bound()),
            prepared_output(crate::execution::structural_state_publication_profiles_encoding_profile(
                workload.model(), workload.procedure_step().unwrap(),
                &[crate::execution::SemanticStatePublicationSource::StructuralRandomStatistics],
                assembly.scope().topology_digest(),
            ).unwrap().encoded_bytes_upper_bound),
            prepared_output_set(),
crate::planning::resource::LocalExecutionPlan::serial(),
)
        .unwrap();
    let FeaResourceDemand::StructuralRandomResponseAssembly(demand) = &assembly_plan.input.demand
    else {
        panic!("random-response assembly plan has the wrong typed demand");
    };
    assert_eq!(
        demand.interval_count.0,
        assembly.scope().plan().interval_count() as u64
    );
    assert_eq!(
        FeaResourcePlan::from_canonical_bytes(&assembly_plan.canonical_bytes().unwrap()).unwrap(),
        assembly_plan
    );
    let mut underreported = assembly_plan.input.clone();
    let FeaResourceDemand::StructuralRandomResponseAssembly(demand) = &mut underreported.demand
    else {
        unreachable!()
    };
    demand.interval_count.0 = demand.task_count.0.saturating_sub(1);
    assert!(build_resource_plan(underreported).is_err());
    let result = StructuralRandomResponseSolverResult::new(&workload, assembly).unwrap();
    assert!(result.canonical_bytes().unwrap().len() as u64 <= result_payload);
    assert_eq!(
        StructuralRandomResponseSolverResult::from_canonical_bytes(
            &result.canonical_bytes().unwrap(),
        )
        .unwrap(),
        result
    );
    (
        result,
        task_workspace_bytes.expect("random-response task workspace"),
    )
}

fn prepared_inputs() -> PreparedInputResourceFacts {
    PreparedInputResourceFacts {
        encoded_payload_bytes: ByteCount(1024),
        transport_closure_bytes: ByteCount(2048),
        decoded_resident_bytes_admission_budget: ByteCount(4096),
    }
}

fn prepared_output(payload: u64) -> PreparedOutputResourceFacts {
    PreparedOutputResourceFacts {
        encoded_payload_bytes_upper_bound: ByteCount(payload),
        transport_closure_bytes_upper_bound: ByteCount(payload + 4096),
        construction_resident_bytes_admission_budget: ByteCount(payload),
    }
}

fn prepared_output_set() -> PreparedOutputSetResourceFacts {
    PreparedOutputSetResourceFacts {
        encoded_bytes_upper_bound: ByteCount(4096),
    }
}

fn integration_policy() -> RandomResponseIntegrationPolicy {
    RandomResponseIntegrationPolicy {
        modal_damping_ratio: 0.04,
        minimum_cumulative_mass_ratio: 0.0,
        relative_variance_tolerance: 1.0e-5,
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
        frequencies_hz: runmat_types::physical::Quantity::new(vec![
            1.0e-3, 100.0, 1_000.0, 10_000.0,
        ]),
        channels: vec![channel],
        spectral_density: runmat_analysis_core::NumericalDataSource::authored(
            vec![
                CrossPowerSpectralDensity::diagonal(&runmat_types::physical::Quantity::new([
                    2.5e-4
                ]));
                4
            ],
            "sha256:test-broadband-base",
        ),
        convention: SpectralDensityConvention::OneSidedPerHertz,
        interpolation: RandomSpectrumInterpolation::LinearFrequency,
        extrapolation: RandomSpectrumExtrapolation::Reject,
        evidence: RandomSpectrumEvidence {
            max_absolute_density_error: (Some(0.0))
                .map(runmat_types::physical::AccelerationPowerSpectralDensity::new),
            equivalent_degrees_of_freedom: None,
        },
    }
}
