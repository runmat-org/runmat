//! Independent uniform-translation reference with signed integral inlet/outlet sources.
//! The unit slab has analytic area one at either end; no solver-derived flow oracle.
use super::fixture::{case, Fixture, ReferenceCase};
use super::reference::rotate;
use runmat_analysis_core::*;

pub fn uniform(n: usize, speed: f64, density: f64, rotated: bool) -> Fixture {
    let mut f = case(n, ReferenceCase::Couette, rotated);
    f.material.density = MassDensity::new(density);
    let velocity = if rotated {
        rotate([speed, 0.0, 0.0])
    } else {
        [speed, 0.0, 0.0]
    };
    let mut ends = [Vec::new(), Vec::new()];
    for face in &f.source.boundary_faces {
        let x = face.centroid_m[usize::from(rotated)];
        if x.abs() < 1e-12 {
            ends[0].extend(face.provenance.clone());
        }
        if (x - 1.0).abs() < 1e-12 {
            ends[1].extend(face.provenance.clone());
        }
    }
    f.constraints.retain_mut(|c| {
        let FluidConstraintKind::PrescribedVelocity { target, value } = &mut c.kind else {
            unreachable!()
        };
        let at_end = ends
            .iter()
            .flatten()
            .any(|e| target.matches_persistent_entity(e));
        *value = PrescribedValue::Constant(velocity.map(Velocity::new));
        !at_end
    });
    f.operators.clear();
    for (i, entities) in ends.into_iter().enumerate() {
        let id = if i == 0 { "mass-inlet" } else { "mass-outlet" };
        let mut target = RegionTarget::boundary(id);
        target.bind_resolved_entities(entities).unwrap();
        if i == 1 {
            f.constraints.push(FluidConstraint {
                constraint_id: "mass-pressure-gauge".into(),
                kind: FluidConstraintKind::PressureReference {
                    target: target.clone(),
                    value: PrescribedValue::Constant(Pressure::new(1e12)),
                },
            });
        }
        f.signals.push(PhysicalSignal::MassFlowRate(Signal {
            signal_id: SignalId(id.into()),
            shape: SignalShape::Scalar,
            frame: None,
            representation: SignalRepresentation::Constant(SignalValue::Scalar(MassFlowRate::new(
                density * speed * if i == 0 { 1.0 } else { -1.0 },
            ))),
            precision: SignalPrecision::Real64,
            evidence: SignalApproximationEvidence {
                max_absolute_error: Some(0.0),
                max_relative_error: Some(0.0),
                derivative_max_absolute_error: None,
                sample_count: 1,
            },
            provenance: SignalProvenance::Authored,
        }));
        f.actions
            .push(ExcitationAction::MassFlow(AnalysisExcitation {
                excitation_id: ExcitationId(id.into()),
                target: ExcitationTarget::Region(target),
                frame: CoordinateFrame::Global,
                direction: None,
                distribution: ExcitationDistribution::PreserveResultant,
                kinematics: ExcitationKinematics::FixedFrame,
                signal_id: SignalId(id.into()),
                active_step_ids: Vec::new(),
                combination_group: None,
                projection: ProjectionRequirement {
                    preserve_resultant: true,
                    preserve_moment: false,
                    relative_tolerance: 1e-12,
                },
            }));
    }
    f
}

/// Exact accelerating translation: grad(p')=-rho*a, no convective or viscous term.
pub fn accelerating(n: usize, start: f64, acceleration: f64, density: f64) -> Fixture {
    let mut f = uniform(n, start, density, false);
    for (i, signal) in f.signals.iter_mut().enumerate() {
        let sign = if i == 0 { 1.0 } else { -1.0 };
        let id = signal.signal_id().0.clone();
        *signal = PhysicalSignal::MassFlowRate(history(
            &id,
            SignalShape::Scalar,
            None,
            [start, start + acceleration]
                .map(|u| SignalValue::Scalar(MassFlowRate::new(sign * density * u))),
        ));
    }
    f.signals.push(PhysicalSignal::Velocity(history(
        "mass-sidewall",
        SignalShape::Vector3,
        Some(CoordinateFrame::Global),
        [start, start + acceleration]
            .map(|u| SignalValue::Vector3([u, 0.0, 0.0].map(Velocity::new))),
    )));
    for c in &mut f.constraints {
        if let FluidConstraintKind::PrescribedVelocity { value, .. } = &mut c.kind {
            *value = PrescribedValue::Signal(SignalId("mass-sidewall".into()));
        }
    }
    f
}

fn history<Q>(
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
