use super::*;
use runmat_analysis_core::*;

fn volume_model() -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let signal_id = SignalId("total_flow".into());
    model.signals = vec![PhysicalSignal::VolumeFlowRate(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::FrequencyHertz,
            knots: [10.0, 30.0]
                .into_iter()
                .map(|frequency| {
                    SignalKnot::continuous(
                        frequency,
                        SignalOrdinate::Complex {
                            real: SignalValue::Scalar(VolumeFlowRate::new(-0.2)),
                            imaginary: SignalValue::Scalar(VolumeFlowRate::new(0.3)),
                        },
                    )
                })
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Complex64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 2,
        },
        provenance: SignalProvenance::Authored,
    })];
    let mut action = model.excitations[0].excitation().clone();
    action.signal_id = signal_id;
    action.distribution = ExcitationDistribution::PreserveResultant;
    action.direction = None;
    action.kinematics = ExcitationKinematics::FixedFrame;
    action.projection.preserve_resultant = true;
    action.projection.preserve_moment = false;
    model.excitations = vec![ExcitationAction::VolumeVelocity(action)];
    model
}

#[test]
fn raw_total_volume_velocity_preserves_complex_quantity_and_rejects_policy_tampering() {
    let model = volume_model();
    let result = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::frequency(20.0),
    )
    .unwrap();
    let entry = result.iter().next().unwrap();
    assert_eq!(entry.quantity, PhysicalQuantityKind::VolumeFlowRate);
    assert!(
        matches!(entry.kind, RealizedExcitationKind::VolumeVelocity(value)
        if value.real == -0.2 && value.imaginary == 0.3)
    );
    for mutation in 0..6 {
        let mut model = volume_model();
        let action = model.excitations[0].excitation_mut();
        match mutation {
            0 => action.projection.preserve_resultant = false,
            1 => action.projection.preserve_moment = true,
            2 => action.kinematics = ExcitationKinematics::Follower,
            3 => {
                action.direction = Some(Direction {
                    vector: [1.0, 0.0, 0.0],
                })
            }
            4 => action.distribution = ExcitationDistribution::UniformKinematic,
            _ => {
                let PhysicalSignal::VolumeFlowRate(signal) = &mut model.signals[0] else {
                    unreachable!()
                };
                signal.shape = SignalShape::Vector3;
            }
        }
        assert!(
            realize_step_excitations(
                &model,
                &model.sequence.steps[0],
                ExcitationSamplePoint::frequency(20.0)
            )
            .is_err(),
            "mutation {mutation}"
        );
    }
}
