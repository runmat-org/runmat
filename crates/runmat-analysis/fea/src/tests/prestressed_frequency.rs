use runmat_analysis_core::{
    ActivationLifecycle, FormulationRequirement, IndependentDomain, ProcedureStep, StateUse,
    StepObjectActivation, StepObjectRef,
};
use runmat_execution::Digest;

use crate::{
    execution::{
        CanonicalSemanticState, CanonicalSemanticStateInput, CanonicalSemanticStatePayload,
        CanonicalStateTransition, SequenceIdentity, SolverResultIdentity, StepIdentity,
        StructuralFrequencyEntryRequest, StructuralPrestressedFrequencySolverTemplate,
        StructuralPrestressedFrequencyTemplate,
    },
    fixtures::{replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind},
    frequency::FrequencyResponsePlan,
    pipeline::structural_frequency::{
        execute_prestressed_structural_frequency_entry, prepare_prestressed_structural_frequency,
    },
    ComputeBackend,
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
    let workload = StructuralPrestressedFrequencyTemplate {
        schema_version: 1,
        model,
        step_id: consumer_id,
        backend: ComputeBackend::Cpu,
        formulation_requirement: FormulationRequirement::Experimental,
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
