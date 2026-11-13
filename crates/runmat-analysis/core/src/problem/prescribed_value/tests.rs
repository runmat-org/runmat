use super::*;
use crate::{
    CartesianFrame, Position, Signal, SignalApproximationEvidence, SignalExtrapolation,
    SignalInterpolation, SignalKnot, SignalOrdinate, SignalPrecision, SignalProvenance,
    SignalShape, TabulatedSignal, TabulatedSignalDomain,
};

fn velocity(
    frame: CoordinateFrame,
    representation: SignalRepresentation<Velocity>,
) -> PhysicalSignal {
    PhysicalSignal::Velocity(Signal {
        signal_id: SignalId("inlet".into()),
        shape: SignalShape::Vector3,
        frame: Some(frame),
        representation,
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: None,
            max_relative_error: None,
            derivative_max_absolute_error: None,
            sample_count: 0,
        },
        provenance: SignalProvenance::Authored,
    })
}

#[test]
fn prescribed_fluid_vector_retains_signed_global_components_and_rejects_scalar_wire() {
    let components = [-2.0, 3.0, -4.0].map(Velocity::new);
    let value = PrescribedValue::Constant(components);
    assert!(value.is_well_formed(&[]));
    assert_eq!(value.study_constant(&[]), Some(&components));
    let wire = serde_json::to_value(&value).unwrap();
    assert_eq!(
        serde_json::from_value::<PrescribedValue<[Velocity; 3]>>(wire).unwrap(),
        value
    );
    let scalar = serde_json::to_value(PrescribedValue::Constant(Velocity::new(2.0))).unwrap();
    assert!(serde_json::from_value::<PrescribedValue<[Velocity; 3]>>(scalar).is_err());
    let invalid = PrescribedValue::Constant([
        Velocity::new(0.0),
        Velocity::new(f64::NAN),
        Velocity::new(0.0),
    ]);
    assert!(!invalid.is_well_formed(&[]));
}

#[test]
fn prescribed_vector_signal_preserves_frame_and_does_not_return_untransformed_local_constant() {
    let components = [-2.0, 3.0, -4.0].map(Velocity::new);
    let global = velocity(
        CoordinateFrame::Global,
        SignalRepresentation::Constant(SignalValue::Vector3(components)),
    );
    let source = PrescribedValue::<[Velocity; 3]>::Signal(SignalId("inlet".into()));
    assert_eq!(
        source.study_constant(std::slice::from_ref(&global)),
        Some(&components)
    );
    let rotated = velocity(
        CoordinateFrame::Cartesian(CartesianFrame {
            frame_id: "rotated".into(),
            origin: Position::new([0.0; 3]),
            x_axis: [0.0, 1.0, 0.0],
            y_axis: [-1.0, 0.0, 0.0],
        }),
        SignalRepresentation::Constant(SignalValue::Vector3(components)),
    );
    let signals = [rotated];
    assert!(source.is_well_formed(&signals));
    assert_eq!(source.signal(&signals), Some(&signals[0]));
    assert_eq!(source.study_constant(&signals), None);
    // Scalar prescribed consumers cannot reinterpret a vector signal as speed.
    assert!(
        !PrescribedValue::<Velocity>::Signal(SignalId("inlet".into())).is_well_formed(&signals)
    );
    assert!(!source.is_well_formed(&[signals[0].clone(), signals[0].clone()]));
}

#[test]
fn prescribed_velocity_table_keeps_each_vector_knot_and_never_samples_implicitly() {
    let signal = velocity(
        CoordinateFrame::Global,
        SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                SignalKnot::continuous(
                    0.0,
                    SignalOrdinate::Real(SignalValue::Vector3([-1.0, 2.0, 0.0].map(Velocity::new))),
                ),
                SignalKnot::continuous(
                    1.0,
                    SignalOrdinate::Real(SignalValue::Vector3([3.0, -4.0, 5.0].map(Velocity::new))),
                ),
            ],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
    );
    let source = PrescribedValue::<[Velocity; 3]>::Signal(SignalId("inlet".into()));
    let signals = [signal];
    assert!(source.is_well_formed(&signals));
    assert_eq!(source.signal(&signals), Some(&signals[0]));
    assert_eq!(source.study_constant(&signals), None);
}

#[test]
fn scalar_prescribed_pressure_contract_remains_scalar() {
    let value = PrescribedValue::Constant(Pressure::new(-5.0));
    assert!(value.is_well_formed(&[]));
    assert_eq!(value.study_constant(&[]), Some(&Pressure::new(-5.0)));
    assert_eq!(<Pressure as PrescribedQuantity>::SHAPE, SignalShape::Scalar);
}

#[test]
fn scalar_velocity_signal_is_not_accepted_as_a_prescribed_vector() {
    let mut scalar = velocity(
        CoordinateFrame::Global,
        SignalRepresentation::Constant(SignalValue::Scalar(Velocity::new(2.0))),
    );
    let PhysicalSignal::Velocity(signal) = &mut scalar else {
        unreachable!();
    };
    signal.shape = SignalShape::Scalar;
    signal.frame = None;
    let signals = [scalar];
    assert!(PrescribedValue::<Velocity>::Signal(SignalId("inlet".into())).is_well_formed(&signals));
    assert!(
        !PrescribedValue::<[Velocity; 3]>::Signal(SignalId("inlet".into()))
            .is_well_formed(&signals)
    );
}
