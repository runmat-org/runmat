use runmat_analysis_core::{
    ActivationLifecycle, IndependentDomain, ProcedureStep, ProcedureStepId, ReferenceStatePolicy,
    SequenceDependency, SequenceDependencyReason, StateId, StateInput, StateSource,
    StateTransition, StateTransitionMethod, StateUse, StepObjectActivation, StepObjectRef,
    TopologyTransitionPolicy, TransitionId,
};
use runmat_execution::Digest;

use crate::{
    execution::{
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        CanonicalStateTransition, SequenceIdentity, SolverResultIdentity, StepIdentity,
        StructuralFrequencyEntryRequest, StructuralPrestressedFrequencySolverTemplate,
        StructuralPrestressedFrequencyTemplate, StructuralPrestressedModalTemplate,
    },
    fixtures::{replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind},
    frequency::FrequencyResponsePlan,
    pipeline::structural_frequency::{
        execute_prestressed_structural_frequency_entry, prepare_prestressed_structural_frequency,
    },
    ComputeBackend, ModalConvergencePolicy, ModalTruncationPolicy,
};

fn direct_case(
    axial_force: f64,
) -> (
    crate::execution::StructuralFrequencyWorkload,
    runmat_meshing_core::SolverMeshArtifact,
    CanonicalSemanticState,
    StructuralFrequencyEntryRequest,
) {
    let (modal_workload, mesh, prior_state) = super::prepared_prestressed_modal_case(axial_force);
    let mut model = modal_workload.model().clone();
    let consumer_id = model.sequence.steps[1].step_id.clone();
    replace_fixture_excitations(
        &mut model,
        vec![
            FixtureExcitation {
                id: "axial_preload".into(),
                region_id: "tip".into(),
                kind: FixtureExcitationKind::Force([0.0, 0.0, axial_force]),
            },
            FixtureExcitation {
                id: "harmonic_probe".into(),
                region_id: "tip".into(),
                kind: FixtureExcitationKind::Force([0.0, 1_000.0, 0.0]),
            },
        ],
    );
    let constraint_activations = model.constraints.iter().map(|constraint| {
        let id = match constraint {
            runmat_analysis_core::AnalysisConstraint::Structural(constraint) => {
                constraint.constraint_id.clone()
            }
            _ => unreachable!(),
        };
        StepObjectActivation {
            object: StepObjectRef::Constraint(id),
            lifecycle: ActivationLifecycle::Activate,
        }
    });
    model.sequence.steps[0].activations = constraint_activations
        .clone()
        .chain([StepObjectActivation {
            object: StepObjectRef::Excitation("axial_preload".into()),
            lifecycle: ActivationLifecycle::Activate,
        }])
        .collect();
    let mut frequency = ProcedureStep::structural_prestressed_harmonic(
        &consumer_id.0,
        "preload",
        "geometric_stiffness",
    );
    frequency.independent_domain = IndependentDomain::FrequencyHertz(vec![60.0]);
    frequency.activations = constraint_activations
        .chain([StepObjectActivation {
            object: StepObjectRef::Excitation("harmonic_probe".into()),
            lifecycle: ActivationLifecycle::Activate,
        }])
        .collect();
    model.sequence.steps[1] = frequency;

    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let producer = StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let source_result = SolverResultIdentity::from_step_result(
        producer,
        prior_state
            .payload
            .structural_linearization()
            .unwrap()
            .identity()
            .digest(),
    )
    .unwrap();
    let state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer,
        declaration: model.sequence.steps[0].state_outputs[1].clone(),
        topology_digest: prior_state.topology_digest,
        source_result,
        payload: CanonicalSemanticStatePayload::StructuralLinearization {
            linearization: Box::new(
                prior_state
                    .payload
                    .structural_linearization()
                    .unwrap()
                    .clone(),
            ),
        },
    })
    .unwrap();
    let transition = CanonicalStateTransition::bind_upstream(
        &model,
        &consumer_id,
        &state,
        StateUse::Prestress,
        Digest::from_bytes(*mesh.canonical_digest.bytes()),
    )
    .unwrap();
    let publication = crate::execution::SemanticStatePublicationRequest::new(
        crate::execution::SemanticStatePublicationSource::StructuralLinearization,
        producer,
        model.sequence.steps[0].state_outputs[1].clone(),
        state.topology_digest,
        source_result,
    )
    .unwrap();
    let profile = publication
        .encoding_profile(
            crate::execution::SemanticStatePublicationProfileSource::StructuralLinearization(
                state.payload.structural_linearization().unwrap(),
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
    let workload = StructuralPrestressedFrequencyTemplate {
        schema_version: 1,
        model,
        step_id: consumer_id,
        backend: ComputeBackend::Cpu,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(60.0, &[], &[]).unwrap(),
        solver: StructuralPrestressedFrequencySolverTemplate::Direct {
            policy: crate::solve::dynamics::DirectHarmonicPolicy::default(),
        },
    }
    .bind(transition, None)
    .unwrap();
    let prepared = prepare_prestressed_structural_frequency(&workload, &mesh, &state).unwrap();
    let request = StructuralFrequencyEntryRequest::new(
        &workload,
        prepared.study,
        prepared.step,
        prepared.execution_plan,
        prepared.scope,
        workload.plan().entries().unwrap()[0],
    )
    .unwrap();
    (workload, mesh, state, request)
}

// witness: FEA-DYNAMICS-PRESTRESS-001/complete_base_operator @
//   prestressed_structural/tet4/numerical_kernel;
//   boundary: Fixed direct entry fixture compares preload response ordering and residual, canonical
//   result roundtrip, and substituted topology rejection. No independent buckling reference,
//   nonlinear producer or generic stability qualification.
#[test]
fn prestressed_direct_harmonic_uses_the_exact_tangent_and_rejects_state_substitution() {
    let solve = |force| {
        let (workload, mesh, state, request) = direct_case(force);
        let result = execute_prestressed_structural_frequency_entry(
            &workload, &mesh, &request, &state, None,
        )
        .unwrap();
        let response_norm = result
            .sample()
            .displacement()
            .real()
            .iter()
            .chain(result.sample().displacement().imaginary())
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        (workload, mesh, state, request, result, response_norm)
    };

    let (_, _, _, _, _, unstressed) = solve(0.0);
    let (_, _, _, _, _, tension) = solve(1.0e4);
    let (workload, mesh, state, request, result, compression) = solve(-1.0e4);
    assert!(compression < unstressed);
    assert!(unstressed < tension);
    assert!(result.evidence().solver_relative_residual() <= 1.0e-9);
    assert_eq!(
        crate::execution::StructuralFrequencyEntryResult::from_canonical_bytes(
            &result.canonical_bytes().unwrap(),
        )
        .unwrap(),
        result
    );

    let mut substituted = state;
    substituted.topology_digest = Digest::sha256(b"substituted-frequency-topology");
    assert!(execute_prestressed_structural_frequency_entry(
        &workload,
        &mesh,
        &request,
        &substituted,
        None,
    )
    .is_err());
}

#[test]
fn complete_prestressed_modal_basis_matches_the_direct_harmonic_solution() {
    let (direct_workload, mesh, prestress_state, direct_request) = direct_case(1.0e4);
    let direct = execute_prestressed_structural_frequency_entry(
        &direct_workload,
        &mesh,
        &direct_request,
        &prestress_state,
        None,
    )
    .unwrap();

    let mut model = direct_workload.model().clone();
    let frequency_step = model.sequence.steps.pop().unwrap();
    let mut modal_step = ProcedureStep::structural_prestressed_modal(
        "prestressed_modes",
        "preload",
        "geometric_stiffness",
    );
    modal_step.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
    let mut modal_frequency_step = frequency_step;
    modal_frequency_step.step_id = ProcedureStepId("prestressed_frequency".into());
    modal_frequency_step.solution_space = runmat_analysis_core::SolutionSpace::ModalReduced;
    modal_frequency_step.state_inputs.push(StateInput {
        input_id: "modal_basis".into(),
        required_kind: runmat_analysis_core::StateKind::ModalBasis,
        source: StateSource::Upstream {
            producer_step: ProcedureStepId("prestressed_modes".into()),
            state_id: StateId("modal_basis".into()),
        },
        use_as: StateUse::ModalBasis,
        conflict: runmat_analysis_core::InitialStateConflictPolicy::Reject,
    });
    model.sequence.steps.push(modal_step.clone());
    model.sequence.steps.push(modal_frequency_step.clone());
    model.sequence.dependencies = vec![
        SequenceDependency {
            predecessor: ProcedureStepId("preload".into()),
            successor: modal_step.step_id.clone(),
            reason: SequenceDependencyReason::StateTransfer,
        },
        SequenceDependency {
            predecessor: ProcedureStepId("preload".into()),
            successor: modal_frequency_step.step_id.clone(),
            reason: SequenceDependencyReason::StateTransfer,
        },
        SequenceDependency {
            predecessor: modal_step.step_id.clone(),
            successor: modal_frequency_step.step_id.clone(),
            reason: SequenceDependencyReason::StateTransfer,
        },
    ];
    model.sequence.transitions.push(StateTransition {
        transition_id: TransitionId("preload-to-prestressed-modal".into()),
        producer_step: ProcedureStepId("preload".into()),
        consumer_step: modal_step.step_id.clone(),
        state_id: StateId("geometric_stiffness".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    model.sequence.transitions[0].consumer_step = modal_frequency_step.step_id.clone();
    model.sequence.transitions[0].transition_id =
        TransitionId("preload-to-prestressed-frequency".into());
    model.sequence.transitions.push(StateTransition {
        transition_id: TransitionId("prestressed-modal-to-frequency".into()),
        producer_step: modal_step.step_id.clone(),
        consumer_step: modal_frequency_step.step_id.clone(),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });

    let topology_digest = Digest::from_bytes(*mesh.canonical_digest.bytes());
    let sequence = SequenceIdentity::from_solver_contract(&model.sequence).unwrap();
    let preload_producer =
        StepIdentity::from_solver_contract(sequence, &model.sequence.steps[0]).unwrap();
    let preload_source = SolverResultIdentity::from_step_result(
        preload_producer,
        prestress_state
            .payload
            .structural_linearization()
            .unwrap()
            .identity()
            .digest(),
    )
    .unwrap();
    let prestress_state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer: preload_producer,
        declaration: model.sequence.steps[0].state_outputs[1].clone(),
        topology_digest,
        source_result: preload_source,
        payload: CanonicalSemanticStatePayload::StructuralLinearization {
            linearization: Box::new(
                prestress_state
                    .payload
                    .structural_linearization()
                    .unwrap()
                    .clone(),
            ),
        },
    })
    .unwrap();
    let modal_transition = CanonicalStateTransition::bind_upstream(
        &model,
        &modal_step.step_id,
        &prestress_state,
        StateUse::Prestress,
        topology_digest,
    )
    .unwrap();
    let modal_workload = StructuralPrestressedModalTemplate {
        schema_version: 1,
        model: model.clone(),
        step_id: modal_step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(3),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
    }
    .bind(modal_transition)
    .unwrap();
    let modal_result = crate::execute_structural_prestressed_modal_workload(
        &modal_workload,
        &mesh,
        &prestress_state,
    )
    .unwrap();
    assert_eq!(modal_result.basis().modes().len(), 3);
    let modal_producer = StepIdentity::from_solver_contract(sequence, &modal_step).unwrap();
    let modal_source = SolverResultIdentity::from_step_result(
        modal_producer,
        modal_result.basis_identity().digest(),
    )
    .unwrap();
    let basis_state = CanonicalSemanticState::new(CanonicalSemanticStateInput {
        producer: modal_producer,
        declaration: modal_step.state_outputs[0].clone(),
        topology_digest,
        source_result: modal_source,
        payload: CanonicalSemanticStatePayload::ModalBasis {
            basis: Box::new(modal_result.basis().clone()),
        },
    })
    .unwrap();
    let frequency_transition = CanonicalStateTransition::bind_upstream(
        &model,
        &modal_frequency_step.step_id,
        &prestress_state,
        StateUse::Prestress,
        topology_digest,
    )
    .unwrap();
    CanonicalStateTransition::bind_upstream(
        &model,
        &modal_frequency_step.step_id,
        &basis_state,
        StateUse::ModalBasis,
        topology_digest,
    )
    .unwrap();
    let modal_workload = StructuralPrestressedFrequencyTemplate {
        schema_version: 1,
        model,
        step_id: modal_frequency_step.step_id,
        backend: ComputeBackend::Cpu,
        observation_execution: crate::results::observation::ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(60.0, &[], &[]).unwrap(),
        solver: StructuralPrestressedFrequencySolverTemplate::ModalReduced,
    }
    .bind(frequency_transition, Some(basis_state.identity))
    .unwrap();
    let prepared =
        prepare_prestressed_structural_frequency(&modal_workload, &mesh, &prestress_state).unwrap();
    let modal_request = StructuralFrequencyEntryRequest::new(
        &modal_workload,
        prepared.study,
        prepared.step,
        prepared.execution_plan,
        prepared.scope,
        modal_workload.plan().entries().unwrap()[0],
    )
    .unwrap();
    let modal = execute_prestressed_structural_frequency_entry(
        &modal_workload,
        &mesh,
        &modal_request,
        &prestress_state,
        Some(&basis_state),
    )
    .unwrap();

    for (direct, modal) in direct
        .sample()
        .displacement()
        .real()
        .iter()
        .chain(direct.sample().displacement().imaginary())
        .zip(
            modal
                .sample()
                .displacement()
                .real()
                .iter()
                .chain(modal.sample().displacement().imaginary()),
        )
    {
        let scale = direct.abs().max(modal.abs()).max(1.0);
        assert!((direct - modal).abs() <= 1.0e-8 * scale);
    }
}
