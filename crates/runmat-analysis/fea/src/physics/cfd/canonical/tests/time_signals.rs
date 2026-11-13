use super::closed_reference::closed;
use super::*;
use crate::excitation::ProjectionSample;
use runmat_analysis_core::*;

pub(super) fn table<Q>(
    id: &str,
    shape: SignalShape,
    frame: Option<CoordinateFrame>,
    values: [SignalValue<Q>; 2],
) -> Signal<Q> {
    Signal {
        signal_id: SignalId(id.into()),
        shape,
        frame,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: [0.0, 1.0]
                .into_iter()
                .zip(values)
                .map(|(t, v)| SignalKnot::continuous(t, SignalOrdinate::Real(v)))
                .collect(),
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 2,
        },
        provenance: SignalProvenance::Authored,
    }
}
fn at(value: f64) -> ProjectionSample {
    ProjectionSample::TimeSeconds {
        value,
        approach: SignalSampleApproach::At,
    }
}

#[test]
fn shared_velocity_and_gauge_histories_advance_original_state_without_absolute_roundtrip() {
    let start = [0.001, 0.002, -0.001];
    let acceleration = [0.01, -0.005, 0.002];
    let end = std::array::from_fn(|k| start[k] + acceleration[k]);
    let mut f = closed(2, start, 0.0);
    f.signals = vec![
        PhysicalSignal::Velocity(table(
            "shared-u",
            SignalShape::Vector3,
            Some(CoordinateFrame::Global),
            [start, end].map(|v| SignalValue::Vector3(v.map(Velocity::new))),
        )),
        PhysicalSignal::Pressure(table(
            "shared-reference",
            SignalShape::Scalar,
            None,
            [0.0, 1e16].map(|v| SignalValue::Scalar(Pressure::new(v))),
        )),
    ];
    for c in &mut f.constraints {
        match &mut c.kind {
            FluidConstraintKind::PrescribedVelocity { value, .. } => {
                *value = PrescribedValue::Signal(SignalId("shared-u".into()))
            }
            FluidConstraintKind::PressureReference { value, .. } => {
                *value = PrescribedValue::Signal(SignalId("shared-reference".into()))
            }
            _ => unreachable!(),
        }
    }
    assert!(
        CanonicalFluidProblem::prepare(f.input()).is_err(),
        "time source cannot silently become steady"
    );
    let initial = CanonicalFluidProblem::prepare_at(f.input(), at(0.0)).unwrap();
    let receiving = CanonicalFluidProblem::prepare_at(f.input(), at(1.0)).unwrap();
    let u = vec![start; f.topology.cells.len()];
    let mut pressure = f
        .topology
        .cells
        .iter()
        .map(|c| -geometry_ops::dot(acceleration, c.centroid_m))
        .collect::<Vec<_>>();
    initial
        .pressure_reference_plan
        .normalize(&initial, &mut pressure)
        .unwrap();
    let q = f
        .topology
        .faces
        .iter()
        .map(|face| geometry_ops::dot(start, face.owner_area_vector_m2))
        .collect::<Vec<_>>();
    let previous = CanonicalFluidFields {
        velocity_m_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Velocity,
            _,
        >::new(&u),
        pressure_fluctuation_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(&pressure),
        pressure_reference_pa: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Pressure,
            _,
        >::new(initial.pressure_reference_pa.canonical_values()),
        face_volume_flux_m3_per_s: runmat_types::physical::Quantity::<
            runmat_types::physical::kind::VolumeFlowRate,
            _,
        >::new(&q),
    };
    assert!(receiving
        .prepare_backward_euler(0.0, 0.5, previous)
        .is_err());
    let interval = receiving
        .prepare_backward_euler(0.0, 1.0, previous)
        .unwrap();
    let result = interval.solve().unwrap();
    for v in result.velocity_m_per_s.canonical_values() {
        assert!(norm(sub(*v, end)) < 1e-8);
    }
    for g in geometry_ops::pressure_gradient(
        &receiving,
        result.pressure_fluctuation_pa.canonical_values(),
    )
    .unwrap()
    {
        assert!(norm(geometry_ops::add(g, acceleration)) < 1e-6);
    }
    assert!(result
        .pressure_pa
        .canonical_values()
        .iter()
        .all(|p| *p == 1e16));
    assert!(result
        .pressure_fluctuation_pa
        .canonical_values()
        .iter()
        .any(|p| p.abs() > 1e-4));
    assert_eq!(
        result.pressure_reference_pa.canonical_values().as_slice(),
        vec![1e16; u.len()]
    );
    let cold: CanonicalFluidResult =
        serde_json::from_slice(&serde_json::to_vec(&result).unwrap()).unwrap();
    assert_eq!(interval.recover(&cold).unwrap(), result.residual);
    assert!(receiving.recover(&cold).is_err());
    let mut corrupt = cold;
    corrupt.pressure_reference_pa = corrupt
        .pressure_reference_pa
        .try_map_canonical_storage(|mut values| {
            values[0] += 2.0;
            values
        })
        .unwrap();
    assert!(interval.recover(&corrupt).is_err());
}

// witness: FEA-FLUID-INPUTS-001/signals @
//   incompressible_laminar_transient/canonical_tetrahedra/numerical_kernel;
//   boundary: Independent volume-integrated time sampling; no partitioned algebra credit.
#[test]
fn body_force_time_sampling_preserves_independent_volume_resultant() {
    let mut f = case(2, ReferenceCase::Duct, false);
    f.signals[0] = PhysicalSignal::BodyForceDensity(table(
        "original-duct-force",
        SignalShape::Vector3,
        Some(CoordinateFrame::Global),
        [[0.0; 3], [1.0, 0.0, 0.0]].map(|v| SignalValue::Vector3(v.map(BodyForceDensity::new))),
    ));
    let p = CanonicalFluidProblem::prepare_at(f.input(), at(0.25)).unwrap();
    let force = f
        .topology
        .cells
        .iter()
        .zip(&p.body_force_n_per_m3)
        .fold([0.0; 3], |sum, (c, v)| {
            geometry_ops::add(sum, geometry_ops::scale(*v, c.volume_m3))
        });
    let volume: f64 = f.topology.cells.iter().map(|c| c.volume_m3).sum();
    assert!(norm(sub(force, [0.25 * volume, 0.0, 0.0])) < 1e-14);
    assert!(p.body_force_n_per_m3.iter().all(|v| *v == [0.25, 0.0, 0.0]));
    assert!(CanonicalFluidProblem::prepare_at(f.input(), at(2.0)).is_err());
    assert!(CanonicalFluidProblem::prepare_at(
        f.input(),
        ProjectionSample::FrequencyHertz {
            value: 0.25,
            approach: SignalSampleApproach::At
        }
    )
    .is_err());
    let mut nonfinite = f.signals.clone();
    let PhysicalSignal::BodyForceDensity(signal) = &mut nonfinite[0] else {
        unreachable!()
    };
    signal.representation = SignalRepresentation::Constant(SignalValue::Vector3([
        BodyForceDensity::new(f64::NAN),
        BodyForceDensity::new(0.0),
        BodyForceDensity::new(0.0),
    ]));
    signal.evidence.sample_count = 1;
    let mut malformed_input = f.input();
    malformed_input.signals = &nonfinite;
    assert!(matches!(
        CanonicalFluidProblem::prepare_at(malformed_input, at(0.25)),
        Err(CanonicalFluidError::InputRealization(_))
    ));
    let mut directed = f.actions.clone();
    let ExcitationAction::BodyForce(action) = &mut directed[0] else {
        unreachable!()
    };
    action.direction = Some(Direction {
        vector: [1.0, 0.0, 0.0],
    });
    let mut directed_input = f.input();
    directed_input.excitations = &directed;
    assert!(CanonicalFluidProblem::prepare_at(directed_input, at(0.25)).is_err());
    let mut bad = f.signals.clone();
    bad.push(bad[0].clone());
    let mut input = f.input();
    input.signals = &bad;
    assert!(CanonicalFluidProblem::prepare_at(input, at(0.25)).is_err());
}

#[test]
fn full_original_signal_payload_is_admitted_before_time_realization() {
    let mut small = closed(2, [0.0; 3], 0.0);
    let base = CanonicalFluidProblem::prepare(small.input())
        .unwrap()
        .reservation();
    let mut signal = table(
        "large-unused-original",
        SignalShape::Vector3,
        Some(CoordinateFrame::Global),
        [[0.0; 3], [0.0; 3]].map(|v| SignalValue::Vector3(v.map(Velocity::new))),
    );
    let SignalRepresentation::Tabulated(t) = &mut signal.representation else {
        unreachable!()
    };
    t.knots = (0..4096)
        .map(|i| {
            SignalKnot::continuous(
                i as f64,
                SignalOrdinate::Real(SignalValue::Vector3([Velocity::new(0.0); 3])),
            )
        })
        .collect();
    signal.evidence.sample_count = 4096;
    small.signals.push(PhysicalSignal::Velocity(signal));
    small.policy.maximum_resident_bytes = Some(base.resident_bytes_upper_bound);
    assert!(matches!(
        CanonicalFluidProblem::prepare_at(small.input(), at(0.0)),
        Err(CanonicalFluidError::Resource(_))
    ));
}

// witness: FEA-FLUID-INPUTS-001/signals @
//   incompressible_laminar_transient/canonical_tetrahedra/numerical_kernel; coverage:
//   partial;
//   boundary: A tabulated vector velocity signal with a one-sided jump is realized on the original
//   boundary faces; pressure, mass-flow and body-force signals are separate tests.
#[test]
fn one_sided_velocity_jump_is_realized_on_original_boundary_faces() {
    let mut f = closed(2, [0.001, 0.0, 0.0], 0.0);
    let mut signal = table(
        "original-jump",
        SignalShape::Vector3,
        Some(CoordinateFrame::Global),
        [[0.001, 0.0, 0.0], [0.002, 0.0, 0.0]].map(|v| SignalValue::Vector3(v.map(Velocity::new))),
    );
    let SignalRepresentation::Tabulated(t) = &mut signal.representation else {
        unreachable!()
    };
    t.knots.insert(
        1,
        SignalKnot::jump(
            0.5,
            SignalOrdinate::Real(SignalValue::Vector3([
                Velocity::new(0.001),
                Velocity::new(0.0),
                Velocity::new(0.0),
            ])),
            SignalOrdinate::Real(SignalValue::Vector3([
                Velocity::new(0.002),
                Velocity::new(0.0),
                Velocity::new(0.0),
            ])),
            SignalPointContinuity::Right,
        ),
    );
    signal.evidence.sample_count = 3;
    f.signals.push(PhysicalSignal::Velocity(signal));
    for c in &mut f.constraints {
        if let FluidConstraintKind::PrescribedVelocity { value, .. } = &mut c.kind {
            *value = PrescribedValue::Signal(SignalId("original-jump".into()));
        }
    }
    let mut identities = Vec::new();
    for (approach, velocity) in [
        (SignalSampleApproach::LeftLimit, 0.001),
        (SignalSampleApproach::RightLimit, 0.002),
    ] {
        let p = CanonicalFluidProblem::prepare_at(
            f.input(),
            ProjectionSample::TimeSeconds {
                value: 0.5,
                approach,
            },
        )
        .unwrap();
        assert!(matches!(
            p.solve(),
            Err(CanonicalFluidError::Unsupported(_))
        ));
        identities.push(p.numerical_input_identity);
        for (face, boundary) in p.topology.faces.iter().zip(&p.boundaries) {
            if face.neighbor.is_none() {
                let boundary::FaceBoundary::Velocity { value, .. } = boundary else {
                    panic!("original physical velocity law")
                };
                assert_eq!(*value, [velocity, 0.0, 0.0]);
            }
        }
    }
    assert_ne!(identities[0], identities[1]);
}
