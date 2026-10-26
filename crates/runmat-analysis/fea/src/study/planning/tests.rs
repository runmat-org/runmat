use runmat_analysis_core::{
    AnalysisSequence, CoordinateFrame, CrossPowerSpectralDensity, IndependentDomain, ProcedureStep,
    RandomExcitationChannel, RandomExcitationChannelId, RandomExcitationId,
    RandomExcitationQuantity, RandomExcitationSpectrum, RandomSpectrumEvidence,
    RandomSpectrumExtrapolation, RandomSpectrumInterpolation, ReferenceStatePolicy,
    SequenceDependencyReason, SpectralDensityConvention, SpectrumDirection, StateId,
    StateTransition, StateTransitionMethod, StructuralRandomResponseSettings,
    TopologyTransitionPolicy, TransitionId,
};

mod thermal;
use runmat_meshing_core::StableDigest;

use crate::{
    execution::{StructuralStepWorkload, StructuralStudyWorkloads},
    fixtures::{fixture_model, FixtureId},
    properties::compile_property_map,
    study::AnalysisStudyCompilationError,
    ComputeBackend,
};

use super::{
    plan_structural_study, CompiledAnalysisStudy, StructuralNumericalPlanningPolicy,
    StructuralStudyPlanningPolicy,
};

#[test]
fn planner_retains_authored_random_response_controls() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.truncate(1);
    let excitation = random_excitation();
    let mut modes = ProcedureStep::structural_modal("modes");
    modes.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
    let response = ProcedureStep::structural_random_response(
        "random",
        excitation.excitation_id.clone(),
        StructuralRandomResponseSettings {
            modal_damping_ratio: 0.037,
            minimum_cumulative_mass_ratio: 0.93,
        },
        "modes",
        "modal_basis",
    );
    model.sequence = state_sequence("random-sequence", modes, response, "modal_basis");
    model.random_excitations = vec![excitation];

    let StructuralStudyWorkloads::HeterogeneousSequence { workloads } =
        plan_structural_study(&compiled(model), policy()).unwrap()
    else {
        panic!("random response should produce a heterogeneous sequence");
    };
    let StructuralStepWorkload::RandomResponse { template } =
        &workloads[&runmat_analysis_core::ProcedureStepId("random".into())]
    else {
        panic!("random response should retain a typed template");
    };
    assert_eq!(template.integration_policy.modal_damping_ratio, 0.037);
    assert_eq!(
        template.integration_policy.minimum_cumulative_mass_ratio,
        0.93
    );
}

#[test]
fn planner_creates_an_unresolved_modal_frequency_consumer() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut modes = ProcedureStep::structural_modal("modes");
    modes.independent_domain = IndependentDomain::Eigenvalue { requested_count: 4 };
    let mut response =
        ProcedureStep::structural_modal_harmonic("frequency", "modes", "modal_basis");
    response.independent_domain = IndependentDomain::FrequencyHertz(vec![12.0, 48.0]);
    model.sequence = state_sequence("modal-frequency", modes, response, "modal_basis");

    let StructuralStudyWorkloads::HeterogeneousSequence { workloads } =
        plan_structural_study(&compiled(model), policy()).unwrap()
    else {
        panic!("modal harmonic should produce a heterogeneous sequence");
    };
    let StructuralStepWorkload::ModalFrequency { template } =
        &workloads[&runmat_analysis_core::ProcedureStepId("frequency".into())]
    else {
        panic!("modal harmonic should remain unresolved until basis publication");
    };
    assert_eq!(template.plan.frequencies_hz(), &[12.0, 48.0]);
    assert_eq!(template.plan.reference_frequency_hz(), 12.0);
}

#[test]
fn compiled_identity_includes_and_validates_the_public_output_contract() {
    let without_output = compiled(fixture_model(FixtureId::CantileverLinearStatic));
    let mut with_output = without_output.clone();
    let step_id = with_output.model.sequence.steps[0].step_id.0.clone();
    with_output
        .outputs
        .push(runmat_analysis_core::AuthoredOutput {
            output_id: "displacement".into(),
            expression: runmat_analysis_core::AuthoredOutputExpression::AnalysisField {
                step_id,
                quantity: runmat_analysis_core::ObservationQuantity::Displacement,
                location: runmat_analysis_core::FieldLocation::Node,
            },
            policy: runmat_analysis_core::AuthoredOutputPolicy::default(),
        });

    assert_ne!(
        without_output.definition_digest().unwrap(),
        with_output.definition_digest().unwrap()
    );
    with_output.outputs[0].expression =
        runmat_analysis_core::AuthoredOutputExpression::AnalysisField {
            step_id: "absent".into(),
            quantity: runmat_analysis_core::ObservationQuantity::Displacement,
            location: runmat_analysis_core::FieldLocation::Node,
        };
    assert!(matches!(
        with_output.definition_digest(),
        Err(AnalysisStudyCompilationError::InvalidOutputs(_))
    ));
}

fn compiled(model: runmat_analysis_core::AnalysisModel) -> CompiledAnalysisStudy {
    let properties = compile_property_map(&model, None).unwrap();
    CompiledAnalysisStudy {
        module_path: "tests/planning.fea".into(),
        study_id: "planning".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        model,
        properties,
        solver_mesh_digest: StableDigest::ZERO,
    }
}

fn policy() -> StructuralStudyPlanningPolicy {
    StructuralStudyPlanningPolicy {
        backend: ComputeBackend::Cpu,
        numerics: StructuralNumericalPlanningPolicy::production(),
    }
}

fn state_sequence(
    id: &str,
    producer: ProcedureStep,
    consumer: ProcedureStep,
    state_id: &str,
) -> AnalysisSequence {
    let mut sequence = AnalysisSequence::from_steps(id, vec![producer.clone(), consumer.clone()]);
    sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    sequence.transitions.push(StateTransition {
        transition_id: TransitionId(format!("{}-to-{}", producer.step_id.0, consumer.step_id.0)),
        producer_step: producer.step_id,
        consumer_step: consumer.step_id,
        state_id: StateId(state_id.into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    sequence
}

fn random_excitation() -> RandomExcitationSpectrum {
    RandomExcitationSpectrum {
        excitation_id: RandomExcitationId("base".into()),
        frequencies_hz: runmat_types::physical::Quantity::new(vec![1.0, 10.0, 100.0]),
        channels: vec![RandomExcitationChannel {
            channel_id: RandomExcitationChannelId("base-x".into()),
            quantity: RandomExcitationQuantity::BaseAcceleration,
            direction: SpectrumDirection {
                frame: CoordinateFrame::Global,
                vector: [1.0, 0.0, 0.0],
            },
        }],
        spectral_density: runmat_analysis_core::NumericalDataSource::authored(
            vec![
                CrossPowerSpectralDensity::diagonal(&runmat_types::physical::Quantity::new([1.0]));
                3
            ],
            "sha256:planning-random",
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
