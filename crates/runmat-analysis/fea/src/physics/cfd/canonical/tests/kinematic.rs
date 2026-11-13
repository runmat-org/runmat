//! Authored `inlet_velocity` and `wall_motion` excitations are the same face law as the
//! equivalent velocity constraints: identical boundaries, identical solution.
use super::*;
use runmat_analysis_core::*;

fn velocity_signal(id: &str, value: SignalValue<Velocity>, shape: SignalShape) -> PhysicalSignal {
    let frame = (shape == SignalShape::Vector3).then_some(CoordinateFrame::Global);
    PhysicalSignal::Velocity(Signal {
        signal_id: SignalId(id.into()),
        shape,
        frame,
        representation: SignalRepresentation::Constant(value),
        precision: SignalPrecision::Real64,
        evidence: SignalApproximationEvidence {
            max_absolute_error: Some(0.0),
            max_relative_error: Some(0.0),
            derivative_max_absolute_error: None,
            sample_count: 1,
        },
        provenance: SignalProvenance::Authored,
    })
}

fn action(id: &str, target: RegionTarget, direction: Option<[f64; 3]>) -> AnalysisExcitation {
    AnalysisExcitation {
        excitation_id: ExcitationId(id.into()),
        target: ExcitationTarget::Region(target),
        frame: CoordinateFrame::Global,
        direction: direction.map(|vector| Direction { vector }),
        distribution: ExcitationDistribution::UniformKinematic,
        kinematics: ExcitationKinematics::FixedFrame,
        signal_id: SignalId(id.into()),
        active_step_ids: Vec::new(),
        combination_group: None,
        projection: ProjectionRequirement {
            preserve_resultant: false,
            preserve_moment: false,
            relative_tolerance: 1e-12,
        },
    }
}

/// Whether a target selects the moving top wall (y = 1) of the Couette slab.
fn is_top_wall(f: &fixture::Fixture, target: &RegionTarget) -> bool {
    f.source.boundary_faces.iter().any(|face| {
        (face.centroid_m[1] - 1.0).abs() < 1e-12
            && face
                .provenance
                .iter()
                .any(|e| target.matches_persistent_entity(e))
    })
}

// witness: FEA-FLUID-INPUTS-001/boundaries @
//   incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel; coverage: partial;
//   boundary: An inlet_velocity and wall_motion constraint pair reproduces constraint-driven
//   Couette flow in the kernel.
// witness: FEA-FLUID-INPUTS-001/signals @ incompressible_laminar_steady/canonical_tetrahedra/numerical_kernel;
//   coverage: partial; boundary: inlet_velocity (vector signal) and wall_motion (scalar speed with
//   unit direction) bind the same face law as velocity constraints and reproduce Couette flow.
#[test]
fn inlet_velocity_and_wall_motion_reproduce_the_constraint_driven_couette_flow() {
    let reference = case(2, ReferenceCase::Couette, false);
    let expected = CanonicalFluidProblem::prepare(reference.input())
        .unwrap()
        .solve()
        .unwrap();
    let mut f = case(2, ReferenceCase::Couette, false);
    let constraints = std::mem::take(&mut f.constraints);
    let mut moving = 0;
    for (i, c) in constraints.into_iter().enumerate() {
        let FluidConstraintKind::PrescribedVelocity {
            target,
            value: PrescribedValue::Constant(value),
        } = &c.kind
        else {
            f.constraints.push(c);
            continue;
        };
        let v = value.map(|x| x.canonical_value());
        let speed = norm(v);
        let id = format!("kinematic-{i}");
        if speed > 0.0 && is_top_wall(&f, target) {
            // A tangential wall speed as a scalar history with a unit direction.
            moving += 1;
            f.signals.push(velocity_signal(
                &id,
                SignalValue::Scalar(Velocity::new(speed)),
                SignalShape::Scalar,
            ));
            f.actions.push(ExcitationAction::WallMotion(action(
                &id,
                target.clone(),
                Some(v.map(|x| x / speed)),
            )));
        } else {
            f.signals.push(velocity_signal(
                &id,
                SignalValue::Vector3(v.map(Velocity::new)),
                SignalShape::Vector3,
            ));
            f.actions.push(ExcitationAction::InletVelocity(action(
                &id,
                target.clone(),
                None,
            )));
        }
    }
    assert!(moving > 0, "the Couette case drives a wall");
    let actual = CanonicalFluidProblem::prepare(f.input())
        .unwrap()
        .solve()
        .unwrap();
    for (u, v) in actual
        .velocity_m_per_s
        .canonical_values()
        .iter()
        .zip(expected.velocity_m_per_s.canonical_values())
    {
        assert!(norm(sub(*u, *v)) < 1e-12);
    }
    for (p, q) in actual
        .pressure_pa
        .canonical_values()
        .iter()
        .zip(expected.pressure_pa.canonical_values())
    {
        assert!((p - q).abs() < 1e-12);
    }
}

#[test]
fn wall_motion_rejects_normal_motion_and_a_non_unit_direction() {
    // The driven Couette wall (nonzero constant velocity) lies in a y plane.
    let moving_wall = |f: &fixture::Fixture| {
        f.constraints
            .iter()
            .find_map(|c| match &c.kind {
                FluidConstraintKind::PrescribedVelocity { target, .. }
                    if is_top_wall(f, target) =>
                {
                    Some(target.clone())
                }
                _ => None,
            })
            .unwrap()
    };
    for direction in [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        let mut g = case(1, ReferenceCase::Couette, false);
        let target = moving_wall(&g);
        g.constraints.retain(|c| {
            !matches!(&c.kind, FluidConstraintKind::PrescribedVelocity { target: t, .. } if *t == target)
        });
        g.signals.push(velocity_signal(
            "wall",
            SignalValue::Scalar(Velocity::new(1.0)),
            SignalShape::Scalar,
        ));
        g.actions.push(ExcitationAction::WallMotion(action(
            "wall",
            target,
            Some(direction),
        )));
        let error = CanonicalFluidProblem::prepare(g.input()).err();
        if direction[0] == 2.0 {
            assert!(
                matches!(
                    error,
                    Some(CanonicalFluidError::Invalid(
                        "fluid kinematic unit direction"
                    ))
                ),
                "{error:?}"
            );
        } else {
            assert!(
                matches!(
                    error,
                    Some(CanonicalFluidError::Unsupported(
                        "normal moving-wall motion requires ALE geometry"
                    ))
                ),
                "{error:?}"
            );
        }
    }
}
