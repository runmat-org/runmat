use crate::*;

fn table(real: f64, imaginary: f64) -> PhysicalSignal {
    PhysicalSignal::VolumeFlowRate(Signal {
        signal_id: SignalId("flow".into()),
        shape: SignalShape::Scalar,
        frame: None,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::FrequencyHertz,
            knots: vec![SignalKnot::continuous(
                10.0,
                SignalOrdinate::Complex {
                    real: SignalValue::Scalar(VolumeFlowRate::new(real)),
                    imaginary: SignalValue::Scalar(VolumeFlowRate::new(imaginary)),
                },
            )],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Complex128,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    })
}

#[test]
fn volume_flow_is_generic_finite_scalar_complex_quantity() {
    let value = table(2.0, -3.0);
    validate_physical_signals(std::slice::from_ref(&value)).unwrap();
    assert_eq!(value.quantity(), PhysicalQuantityKind::VolumeFlowRate);
    assert_eq!(value.domain(), SignalDomainKind::Frequency);
    assert_eq!(value.precision(), SignalPrecision::Complex128);
    assert_eq!(value.signal_id(), &SignalId("flow".into()));
    assert_eq!(value.provenance(), &SignalProvenance::Authored);
    let encoded = serde_json::to_value(&value).unwrap();
    assert!(encoded.get("volume_flow_rate").is_some());
    assert_eq!(
        serde_json::from_value::<PhysicalSignal>(encoded).unwrap(),
        value
    );
    assert_eq!(
        VolumeFlowRate::from_canonical_value(-2.5).canonical_value(),
        -2.5
    );
    for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(validate_physical_signals(&[table(nonfinite, 0.0)]).is_err());
        assert!(validate_physical_signals(&[table(0.0, nonfinite)]).is_err());
    }
}

#[test]
fn volume_flow_rejects_vector_shapes_and_scalar_frames() {
    let value = table(1.0, 0.0);
    assert!(validate_physical_signals(&[value.with_shape(SignalShape::Vector3)]).is_err());
    let PhysicalSignal::VolumeFlowRate(mut signal) = value else {
        unreachable!()
    };
    signal.frame = Some(CoordinateFrame::Global);
    assert!(validate_physical_signals(&[PhysicalSignal::VolumeFlowRate(signal)]).is_err());
}
