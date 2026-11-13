use super::*;
use crate::{
    CoordinateFrame, ExcitationId, HeatFlux, ProjectionRequirement, RegionTarget, Signal,
    SignalApproximationEvidence, SignalPrecision, SignalProvenance, SignalRepresentation,
    SignalSampleApproach, SignalValue, Velocity,
};

#[test]
fn procedure_case_selection_requires_unique_known_cases_but_permits_empty() {
    use crate::{ExcitationCase, ExcitationCaseId, ExcitationCaseSelection, ProcedureStep};
    let id = ExcitationCaseId("probe".into());
    let cases = [ExcitationCase {
        case_id: id.clone(),
        action_ids: vec![ExcitationId("force".into())],
        activation: ExcitationCaseActivation::Always,
    }];
    let mut step = ProcedureStep::structural_static("response");
    assert_eq!(step.excitation_case_selection, ExcitationCaseSelection::All);
    assert!(validate_case_selections(&cases, &[step.clone()]).is_ok());
    for selected in [vec![], vec![id.clone()]] {
        step.excitation_case_selection = ExcitationCaseSelection::Only(selected);
        assert!(validate_case_selections(&cases, &[step.clone()]).is_ok());
    }
    for selected in [
        vec![id.clone(), id],
        vec![ExcitationCaseId("missing".into())],
    ] {
        step.excitation_case_selection = ExcitationCaseSelection::Only(selected);
        assert!(matches!(
            validate_case_selections(&cases, &[step.clone()]),
            Err(AnalysisValidationError::InvalidProcedureStep { .. })
        ));
    }
}

#[test]
fn scalar_integral_action_rejects_a_moment_requirement() {
    let signal = PhysicalSignal::HeatFlux(Signal {
        signal_id: SignalId("flux".to_string()),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Constant(SignalValue::Scalar(HeatFlux::new(10.0))),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    });
    let excitation = AnalysisExcitation {
        excitation_id: ExcitationId("flux".to_string()),
        target: ExcitationTarget::Region(RegionTarget::boundary("wall")),
        frame: CoordinateFrame::Global,
        direction: None,
        distribution: ExcitationDistribution::IntensityPerArea,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id: SignalId("flux".to_string()),
        active_step_ids: Vec::new(),
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: true,
            preserve_moment: true,
            relative_tolerance: 1.0e-12,
        },
    };

    let action = ExcitationAction::HeatFlux(excitation);
    assert_eq!(
        validate_action_policy(&action, action.excitation(), &signal),
        Err("a scalar-integral excitation cannot require moment conservation".to_string())
    );
}

#[test]
fn initial_signal_sample_rejects_noncanonical_negative_zero_abscissa() {
    let condition = InitialConditionValue {
        initial_condition_id: "initial-velocity".into(),
        target: InitialConditionTarget::WholeModel,
        frame: CoordinateFrame::Global,
        value: InitialValue::SignalSample {
            signal_id: SignalId("velocity-history".into()),
            abscissa: -0.0,
            approach: SignalSampleApproach::At,
        },
        applies_before_step_id: "transient".into(),
    };
    let signals = HashMap::new();
    let steps = HashSet::from(["transient"]);

    assert_eq!(
        validate_initial::<Velocity>(
            &condition,
            PhysicalQuantityKind::Velocity,
            InitialValueShape::Vector3,
            &signals,
            &steps,
        ),
        Err("initial-condition value is not finite or complete".into())
    );
}
