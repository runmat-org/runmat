use super::*;
use crate::{BodyForceDensity, CoordinateFrame, FieldLocation, Temperature};

fn field<Q>(width: usize, representation: SignalRepresentation<Q>, samples: usize) -> Signal<Q> {
    Signal {
        signal_id: SignalId("mesh_field".into()),
        shape: SignalShape::TopologyField {
            topology_id: "mesh_digest".into(),
            location: FieldLocation::Node,
            component_count: width,
        },
        frame: Some(CoordinateFrame::Global),
        representation,
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: samples,
        },
        provenance: SignalProvenance::Authored,
    }
}

fn body(rows: usize, width: usize) -> SignalValue<BodyForceDensity> {
    SignalValue::Tensor {
        shape: vec![rows, width],
        values: (0..rows * width)
            .map(|i| BodyForceDensity::new(i as f64))
            .collect(),
    }
}

fn table(values: Vec<SignalValue<BodyForceDensity>>) -> Signal<BodyForceDensity> {
    let count = values.len();
    field(
        3,
        SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: values
                .into_iter()
                .enumerate()
                .map(|(i, value)| SignalKnot::continuous(i as f64, SignalOrdinate::Real(value)))
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        count,
    )
}

fn waveform() -> Signal<BodyForceDensity> {
    field(
        3,
        SignalRepresentation::Waveform(StandardWaveform {
            kind: StandardWaveformKind::Sine,
            amplitude: body(2, 3),
            offset: Some(body(2, 3)),
            start_seconds: runmat_types::physical::Time::new(0.0),
            duration_seconds: (None).map(runmat_types::physical::Time::new),
            frequency_hz: Some(runmat_types::physical::Frequency::new(2.0)),
            phase_radians: runmat_types::physical::Angle::new(0.0),
            duty_cycle: None,
        }),
        0,
    )
}

#[test]
fn typed_node_vector_and_scalar_concrete_fields_are_valid() {
    let density = field(3, SignalRepresentation::Constant(body(2, 3)), 1);
    let scalar = field(
        1,
        SignalRepresentation::Constant(SignalValue::Tensor {
            shape: vec![2, 1],
            values: vec![Temperature::new(300.0), Temperature::new(310.0)],
        }),
        1,
    );
    assert!(validate_signals(&[PhysicalSignal::BodyForceDensity(density)]).is_ok());
    assert!(validate_signals(&[PhysicalSignal::Temperature(scalar)]).is_ok());
    assert!(validate_signal(&table(vec![body(2, 3), body(2, 3)]), None).is_ok());
    assert!(validate_signal(&waveform(), None).is_ok());
}

#[test]
fn scalar_topology_values_do_not_require_a_component_frame() {
    let mut scalar = field(
        1,
        SignalRepresentation::Constant(SignalValue::Tensor {
            shape: vec![2, 1],
            values: vec![Temperature::new(300.0), Temperature::new(310.0)],
        }),
        1,
    );
    scalar.frame = None;
    assert!(validate_signal(&scalar, None).is_ok());
    let mut vector = field(3, SignalRepresentation::Constant(body(2, 3)), 1);
    vector.frame = None;
    assert!(validate_signal(&vector, None).is_err());
}

#[test]
fn concrete_fields_require_exact_finite_positive_two_dimensional_tensors() {
    let invalid = [
        body(0, 3),
        body(2, 2),
        SignalValue::Vector3([BodyForceDensity::new(1.0); 3]),
        SignalValue::Tensor {
            shape: vec![6],
            values: vec![BodyForceDensity::new(1.0); 6],
        },
        SignalValue::Tensor {
            shape: vec![2, 3, 1],
            values: vec![BodyForceDensity::new(1.0); 6],
        },
        SignalValue::Tensor {
            shape: vec![2, 3],
            values: vec![BodyForceDensity::new(1.0); 5],
        },
        SignalValue::Tensor {
            shape: vec![2, 3],
            values: vec![BodyForceDensity::new(f64::NAN); 6],
        },
        SignalValue::Tensor {
            shape: vec![2, 3],
            values: vec![BodyForceDensity::new(f64::INFINITY); 6],
        },
        SignalValue::Tensor {
            shape: vec![usize::MAX, 3],
            values: Vec::new(),
        },
    ];
    for value in invalid {
        assert!(
            validate_signal(&field(3, SignalRepresentation::Constant(value), 1), None).is_err()
        );
    }
    let mut signal = field(3, SignalRepresentation::Constant(body(2, 3)), 1);
    signal.shape = SignalShape::TopologyField {
        topology_id: "".into(),
        location: FieldLocation::Node,
        component_count: 3,
    };
    assert!(validate_signal(&signal, None).is_err());
    signal.shape = SignalShape::TopologyField {
        topology_id: "mesh_digest".into(),
        location: FieldLocation::Node,
        component_count: 0,
    };
    assert!(validate_signal(&signal, None).is_err());
}

#[test]
fn all_table_ordinates_and_waveform_offsets_share_the_inferred_row_count() {
    assert!(validate_signal(&table(vec![body(2, 3), body(3, 3)]), None).is_err());
    assert!(validate_signal(&table(vec![body(2, 3), body(2, 2)]), None).is_err());
    assert!(validate_signal(&table(Vec::new()), None).is_err());
    let mut signal = table(vec![body(2, 3), body(2, 3), body(2, 3)]);
    let SignalRepresentation::Tabulated(data) = &mut signal.representation else {
        unreachable!()
    };
    data.knots[1] = SignalKnot::jump(
        1.0,
        SignalOrdinate::Real(body(2, 3)),
        SignalOrdinate::Real(body(3, 3)),
        SignalPointContinuity::Right,
    );
    assert!(validate_signal(&signal, None).is_err());
    let mut signal = waveform();
    let SignalRepresentation::Waveform(data) = &mut signal.representation else {
        unreachable!()
    };
    data.offset = Some(body(3, 3));
    assert!(validate_signal(&signal, None).is_err());
    let SignalRepresentation::Waveform(data) = &mut signal.representation else {
        unreachable!()
    };
    data.offset = Some(body(2, 2));
    assert!(validate_signal(&signal, None).is_err());
    let nonfinite = SignalValue::Tensor {
        shape: vec![2, 3],
        values: vec![BodyForceDensity::new(f64::NAN); 6],
    };
    assert!(validate_signal(&table(vec![body(2, 3), nonfinite.clone()]), None).is_err());
    let SignalRepresentation::Waveform(data) = &mut signal.representation else {
        unreachable!()
    };
    data.offset = Some(nonfinite);
    assert!(validate_signal(&signal, None).is_err());
}

#[test]
fn complex_topology_tables_preserve_precision_and_component_consistency() {
    let mut signal = table(vec![body(2, 3), body(2, 3)]);
    let SignalRepresentation::Tabulated(data) = &mut signal.representation else {
        unreachable!()
    };
    for knot in &mut data.knots {
        knot.value = SignalKnotValue::Continuous(SignalOrdinate::Complex {
            real: body(2, 3),
            imaginary: body(2, 3),
        });
    }
    assert!(validate_signal(&signal, None).is_err());
    signal.precision = SignalPrecision::Complex128;
    assert!(validate_signal(&signal, None).is_ok());
    let SignalRepresentation::Tabulated(data) = &mut signal.representation else {
        unreachable!()
    };
    data.knots[1].value = SignalKnotValue::Continuous(SignalOrdinate::Complex {
        real: body(2, 3),
        imaginary: body(3, 3),
    });
    assert!(validate_signal(&signal, None).is_err());
}

// witness: FEA-SHARED-EXCITATION-001/dimensioned_signal_contract @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel; coverage: partial;
//   boundary: Field validation preserves frames, precision and approximation evidence.
#[test]
fn field_validation_preserves_frames_precision_and_approximation_evidence() {
    let mut signal = field(3, SignalRepresentation::Constant(body(2, 3)), 1);
    signal.frame = None;
    assert!(validate_signal(&signal, None).is_err());
    signal.frame = Some(CoordinateFrame::Global);
    signal.precision = SignalPrecision::Complex128;
    assert!(validate_signal(&signal, None).is_err());
    signal.precision = SignalPrecision::Real64;
    signal.evidence.sample_count = 2;
    assert!(validate_signal(&signal, None).is_err());
    signal.evidence.sample_count = 1;
    signal.evidence.max_relative_error = Some(-1.0);
    assert!(validate_signal(&signal, None).is_err());
    signal.evidence.max_relative_error = Some(f64::INFINITY);
    assert!(validate_signal(&signal, None).is_err());
    let mut scalar = field(
        1,
        SignalRepresentation::Constant(SignalValue::Tensor {
            shape: vec![1, 1],
            values: vec![Temperature::new(300.0)],
        }),
        1,
    );
    scalar.frame = None;
    assert!(validate_signal(&scalar, None).is_ok());
    scalar.evidence.max_absolute_error = Some(f64::NAN);
    assert!(validate_signal(&scalar, None).is_err());
}

#[test]
fn topology_artifacts_still_bind_the_declared_digest_and_location_is_explicit() {
    let mut signal = field::<BodyForceDensity>(
        3,
        SignalRepresentation::Artifact(SignalArtifactRef {
            artifact_id: "field_artifact".into(),
            content_digest: "content".into(),
            schema: "field_v1".into(),
            domain: SignalArtifactDomain::Constant,
            topology_digest: Some("mesh_digest".into()),
            sample_count: 1,
        }),
        1,
    );
    assert!(validate_signal(&signal, None).is_ok());
    let SignalRepresentation::Artifact(artifact) = &mut signal.representation else {
        unreachable!()
    };
    artifact.topology_digest = Some("foreign_mesh".into());
    assert!(validate_signal(&signal, None).is_err());
    let SignalRepresentation::Artifact(artifact) = &mut signal.representation else {
        unreachable!()
    };
    artifact.topology_digest = None;
    assert!(validate_signal(&signal, None).is_err());
    let node = serde_json::to_value(&signal.shape).unwrap();
    let mut cell_shape = signal.shape.clone();
    let SignalShape::TopologyField { location, .. } = &mut cell_shape else {
        unreachable!()
    };
    *location = FieldLocation::Cell;
    assert_ne!(node, serde_json::to_value(cell_shape).unwrap());
    let mut missing_location = node;
    missing_location["topology_field"]
        .as_object_mut()
        .unwrap()
        .remove("location");
    assert!(serde_json::from_value::<SignalShape>(missing_location).is_err());
}

#[test]
fn ordinary_scalar_vector_and_tensor_shape_validation_is_unchanged() {
    let mut signal = field(3, SignalRepresentation::Constant(body(2, 3)), 1);
    signal.shape = SignalShape::Tensor(vec![2, 3]);
    assert!(validate_signal(&signal, None).is_ok());
    signal.shape = SignalShape::Vector3;
    signal.representation =
        SignalRepresentation::Constant(SignalValue::Vector3([BodyForceDensity::new(1.0); 3]));
    assert!(validate_signal(&signal, None).is_ok());
    signal.shape = SignalShape::Scalar;
    signal.representation =
        SignalRepresentation::Constant(SignalValue::Scalar(BodyForceDensity::new(1.0)));
    assert!(validate_signal(&signal, None).is_err());
    signal.frame = None;
    assert!(validate_signal(&signal, None).is_ok());
}
