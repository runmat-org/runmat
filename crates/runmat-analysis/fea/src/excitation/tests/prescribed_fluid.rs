use super::*;
use crate::excitation::{sample_prescribed_pressure, sample_prescribed_velocity, ProjectionSample};
use runmat_analysis_core::*;

fn jump<Q: Clone>(
    id: &str,
    time: f64,
    shape: SignalShape,
    frame: Option<CoordinateFrame>,
    left: SignalValue<Q>,
    right: SignalValue<Q>,
) -> Signal<Q> {
    Signal {
        signal_id: SignalId(id.into()),
        shape,
        frame,
        representation: SignalRepresentation::Tabulated(TabulatedSignal {
            domain: TabulatedSignalDomain::TimeSeconds,
            knots: vec![
                SignalKnot::continuous(0.0, SignalOrdinate::Real(left.clone())),
                SignalKnot::jump(
                    time,
                    SignalOrdinate::Real(left),
                    SignalOrdinate::Real(right.clone()),
                    SignalPointContinuity::Right,
                ),
                SignalKnot::continuous(1.0, SignalOrdinate::Real(right)),
            ],
            interpolation: SignalInterpolation::Linear,
            extrapolation: SignalExtrapolation::Reject,
        }),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 3,
        },
        provenance: SignalProvenance::Authored,
    }
}
fn pressure(id: &str, time: f64) -> PhysicalSignal {
    PhysicalSignal::Pressure(jump(
        id,
        time,
        SignalShape::Scalar,
        None,
        SignalValue::Scalar(Pressure::new(1.0)),
        SignalValue::Scalar(Pressure::new(2.0)),
    ))
}
fn velocity(id: &str, time: f64) -> PhysicalSignal {
    PhysicalSignal::Velocity(jump(
        id,
        time,
        SignalShape::Vector3,
        Some(CoordinateFrame::Global),
        SignalValue::Vector3([Velocity::new(1.0); 3]),
        SignalValue::Vector3([Velocity::new(2.0); 3]),
    ))
}
#[test]
fn prescribed_scalar_rejects_duplicate_identity_and_keeps_jump_sides() {
    let signal = pressure("original-pressure", 0.5);
    let value = PrescribedValue::Signal(SignalId("original-pressure".into()));
    for (approach, expected) in [
        (SignalSampleApproach::LeftLimit, 1.0),
        (SignalSampleApproach::RightLimit, 2.0),
        (SignalSampleApproach::At, 2.0),
    ] {
        let point = ExcitationSamplePoint::TimeSeconds {
            value: 0.5,
            approach,
        };
        assert_eq!(
            sample_prescribed_pressure(&value, std::slice::from_ref(&signal), point)
                .unwrap()
                .real,
            expected
        );
    }
    assert!(sample_prescribed_pressure(
        &value,
        &[signal.clone(), signal],
        ExcitationSamplePoint::time(0.5)
    )
    .is_err());
}
#[test]
fn shared_time_breakpoints_include_fluid_gauge_and_exact_outlet_boundaries() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.excitations.clear();
    model.excitation_cases.clear();
    model.constraints.clear();
    model.boundaries.clear();
    model.constraints = vec![
        AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "gauge".into(),
            kind: FluidConstraintKind::PressureReference {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "gauge-face",
                )),
                value: PrescribedValue::Signal(SignalId("gauge-source".into())),
            },
        }),
        AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "velocity".into(),
            kind: FluidConstraintKind::PrescribedVelocity {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "inlet-face",
                )),
                value: PrescribedValue::Signal(SignalId("velocity-source".into())),
            },
        }),
        AnalysisConstraint::Fluid(FluidConstraint {
            constraint_id: "moving-wall".into(),
            kind: FluidConstraintKind::MovingWall {
                target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                    "wall-face",
                )),
                velocity: PrescribedValue::Signal(SignalId("wall-source".into())),
            },
        }),
    ];
    model.boundaries.push(AnalysisBoundary {
        boundary_id: "outlet".into(),
        target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face("outlet-face")),
        operator: BoundaryOperator::FluidOutletPressure {
            value: PrescribedValue::Signal(SignalId("outlet-source".into())),
        },
    });
    model.signals = vec![
        pressure("gauge-source", 0.25),
        pressure("outlet-source", 0.75),
        velocity("velocity-source", 0.4),
        velocity("wall-source", 0.6),
    ];
    let step = ProcedureStep::fluid_transient("transient");
    assert_eq!(
        crate::excitation::time_breakpoints(&model, &step, 0.0, 1.0).unwrap(),
        vec![0.25, 0.4, 0.6, 0.75]
    );
    // This is a signal-selection test, not an admitted physical gauge+outlet model.
}
#[test]
fn canonical_sample_roundtrip_and_prescribed_vector_rejection_are_shared() {
    for sample in [
        ProjectionSample::Static,
        ProjectionSample::TimeSeconds {
            value: 0.5,
            approach: SignalSampleApproach::LeftLimit,
        },
    ] {
        let point = ExcitationSamplePoint::try_from(&sample).unwrap();
        assert_eq!(ProjectionSample::from(point), sample);
    }
    for value in [f64::NAN, f64::INFINITY, -0.0] {
        assert!(
            ExcitationSamplePoint::try_from(&ProjectionSample::TimeSeconds {
                value,
                approach: SignalSampleApproach::At
            })
            .is_err()
        );
    }
    assert!(sample_prescribed_velocity(
        &PrescribedValue::Constant([Velocity::new(f64::INFINITY); 3]),
        &[],
        ExcitationSamplePoint::time(0.0)
    )
    .is_err());
}
