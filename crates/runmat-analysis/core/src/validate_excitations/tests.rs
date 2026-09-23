use super::*;
use crate::{
    CoordinateFrame, ExcitationId, HeatFlux, ProjectionRequirement, RegionTarget, Signal,
    SignalApproximationEvidence, SignalPrecision, SignalProvenance, SignalRepresentation,
    SignalSampleApproach, SignalValue, Velocity,
};

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
