use super::*;
use runmat_analysis_core::*;

fn incident_model() -> AnalysisModel {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let signal_id = SignalId("incident.pressure".into());
    model.signals = vec![PhysicalSignal::Pressure(Signal {
        signal_id: signal_id.clone(),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::FrequencyHertz,
            knots: [(10.0, [1.0, -2.0]), (30.0, [5.0, 6.0])]
                .into_iter()
                .map(|(frequency, p)| {
                    SignalKnot::continuous(
                        frequency,
                        SignalOrdinate::Complex {
                            real: SignalValue::Scalar(Pressure::new(p[0])),
                            imaginary: SignalValue::Scalar(Pressure::new(p[1])),
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
    action.distribution = ExcitationDistribution::UniformKinematic;
    action.direction = None;
    action.projection.preserve_moment = false;
    model.excitations = vec![ExcitationAction::AcousticIncidentPressure(action)];
    model
}

#[test]
fn raw_incident_realization_cannot_ignore_authored_policy_or_shape() {
    for mutation in 0..5 {
        let mut model = incident_model();
        let action = model.excitations[0].excitation_mut();
        match mutation {
            0 => {
                action.direction = Some(Direction {
                    vector: [1.0, 0.0, 0.0],
                })
            }
            1 => action.distribution = ExcitationDistribution::IntensityPerArea,
            2 => action.kinematics = ExcitationKinematics::Follower,
            3 => action.projection.preserve_moment = true,
            _ => {
                let PhysicalSignal::Pressure(signal) = &mut model.signals[0] else {
                    unreachable!()
                };
                signal.shape = SignalShape::Vector3; // Payload still scalar: declaration must matter.
            }
        }
        assert!(realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            ExcitationSamplePoint::frequency(20.0)
        )
        .is_err());
    }
}

#[test]
fn shared_incident_pressure_spectrum_preserves_phase_identity_and_components() {
    let model = incident_model();
    let realized = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::frequency(20.0),
    )
    .unwrap();
    let entry = realized.iter().next().unwrap();
    assert_eq!(realized.len(), 1);
    assert_eq!(entry.signal_id, SignalId("incident.pressure".into()));
    assert_eq!(
        entry.excitation_id,
        model.excitations[0].excitation().excitation_id
    );
    assert_eq!(entry.quantity, PhysicalQuantityKind::Pressure);
    assert!(
        matches!(entry.kind, RealizedExcitationKind::AcousticIncidentPressure(value) if value.real == 3.0 && value.imaginary == 2.0)
    );
    for (component, expected) in [
        (super::super::ComplexComponent::Real, 3.0),
        (super::super::ComplexComponent::Imaginary, 2.0),
    ] {
        assert!(
            matches!(realized.component(component).iter().next().unwrap().kind,
            RealizedExcitationKind::AcousticIncidentPressure(value) if value.real == expected && value.imaginary == 0.0)
        );
    }
    assert!(realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::frequency(40.0)
    )
    .is_err());
}

#[test]
fn incident_pressure_rejects_vector_and_malformed_scalar_shapes() {
    for shape in [SignalShape::Scalar, SignalShape::Vector3] {
        let mut model = incident_model();
        let PhysicalSignal::Pressure(signal) = &mut model.signals[0] else {
            unreachable!()
        };
        signal.shape = shape;
        let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
            unreachable!()
        };
        for knot in &mut table.knots {
            knot.value = SignalKnotValue::Continuous(SignalOrdinate::Complex {
                real: SignalValue::Vector3([Pressure::new(1.0); 3]),
                imaginary: SignalValue::Vector3([Pressure::new(2.0); 3]),
            });
        }
        assert!(realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            ExcitationSamplePoint::frequency(20.0)
        )
        .is_err());
    }
}

#[test]
fn incident_pressure_uses_shared_quantity_sampler_and_rejects_nonfinite_pressure() {
    let mut model = incident_model();
    let wrong = fixture_model(FixtureId::CantileverLinearStatic)
        .signals
        .remove(0);
    assert!(super::super::action::realize_kind(
        &model.excitations[0],
        &wrong,
        ExcitationSamplePoint::frequency(20.0)
    )
    .is_err());
    let PhysicalSignal::Pressure(signal) = &mut model.signals[0] else {
        unreachable!()
    };
    let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
        unreachable!()
    };
    table.knots[0].value = SignalKnotValue::Continuous(SignalOrdinate::Complex {
        real: SignalValue::Scalar(Pressure::new(f64::INFINITY)),
        imaginary: SignalValue::Scalar(Pressure::new(0.0)),
    });
    assert!(realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::frequency(20.0)
    )
    .is_err());
}

#[test]
fn shared_scalar_guard_rejects_each_component_and_interpolation_overflow() {
    use super::super::sample::{expect_scalar, sample_prescribed_pressure, SampledValue};
    use super::super::ComplexScalar;
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (real, imaginary) in [(invalid, 0.0), (0.0, invalid)] {
            assert!(
                expect_scalar(SampledValue::Scalar(ComplexScalar { real, imaginary })).is_err()
            );
        }
        assert!(sample_prescribed_pressure(
            &PrescribedValue::Constant(Pressure::new(invalid)),
            &[],
            ExcitationSamplePoint::frequency(20.0),
        )
        .is_err());
    }
    let mut model = incident_model();
    let PhysicalSignal::Pressure(signal) = &mut model.signals[0] else {
        unreachable!()
    };
    let SignalRepresentation::Tabulated(table) = &mut signal.representation else {
        unreachable!()
    };
    // Both endpoints finite; the existing interpolation law overflows b-a.
    for (knot, value) in table.knots.iter_mut().zip([-f64::MAX, f64::MAX]) {
        knot.value = SignalKnotValue::Continuous(SignalOrdinate::Complex {
            real: SignalValue::Scalar(Pressure::new(value)),
            imaginary: SignalValue::Scalar(Pressure::new(0.0)),
        });
    }
    assert!(realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::frequency(20.0)
    )
    .is_err());
}
