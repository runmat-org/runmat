use super::*;
use crate::topology::BoundarySurface;
use runmat_meshing_core::BoundaryTriangleOrder;
// witness: FEA-CONTACT-OPERATOR-001/references @ penalty_coulomb_contact/tet4/numerical_kernel;
//   boundary: Tri3 contact frame and penalty friction point-law proper-rotation objectivity; no
//   global Hertz, sliding body, action-reaction or mesh/penalty convergence credit.
#[test]
fn material_attached_frame_and_point_traction_are_objective_under_proper_rotation() {
    let original = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let rotate = |p: [f64; 3]| [p[2], p[0], p[1]];
    let shift = |p: [f64; 3]| {
        let r = rotate(p);
        [r[0] + 3.0, r[1] - 2.0, r[2] + 4.0]
    };
    let a = BoundarySurface::derive(BoundaryTriangleOrder::Tri3, &original)
        .unwrap()
        .evaluate([0.5, 0.2, 0.3])
        .unwrap();
    let b = BoundarySurface::derive(BoundaryTriangleOrder::Tri3, &original.map(shift))
        .unwrap()
        .evaluate([0.5, 0.2, 0.3])
        .unwrap();
    let frame_a = ContactPointFrame::from_surface(&a).unwrap();
    let frame_b = ContactPointFrame::from_surface(&b).unwrap();
    let separation = [0.01, -0.02, -0.03];
    let motion = |v: [f64; 3]| {
        runmat_types::physical::Quantity::<runmat_types::physical::kind::Length, _>::new(v)
    };
    let rigid_increment = frame_b
        .objective_tangential_increment(motion(rotate(separation)), &frame_a, motion(separation))
        .unwrap();
    for value in rigid_increment.canonical_value() {
        near(value, 0.0);
    }
    let with_slip = [separation[0] + 0.005, separation[1] - 0.002, separation[2]];
    let material_increment = frame_b
        .objective_tangential_increment(motion(rotate(with_slip)), &frame_a, motion(separation))
        .unwrap()
        .canonical_value();
    near(material_increment[0], 0.005);
    near(material_increment[1], -0.002);
    let increment = [0.08, 0.06, 0.0];
    let local_a = frame_a.tangential_increment(motion(increment)).unwrap();
    let local_b = frame_b
        .tangential_increment(motion(rotate(increment)))
        .unwrap();
    for i in 0..2 {
        near(local_a.canonical_value()[i], local_b.canonical_value()[i]);
    }
    let law = law(0.3);
    let history = law
        .update(
            runmat_types::physical::Length::new(-0.01),
            runmat_types::physical::Quantity::new([0.01, 0.0]),
            &law.initial_state(),
        )
        .unwrap()
        .trial_state;
    let response_a = law
        .update(
            runmat_types::physical::Length::new(-0.01),
            local_a,
            &history,
        )
        .unwrap();
    let response_b = law
        .update(
            runmat_types::physical::Length::new(-0.01),
            local_b,
            &history,
        )
        .unwrap();
    assert_eq!(response_a.trial_state, response_b.trial_state);
    let force_a = rotate(
        frame_a
            .internal_traction_world(&response_a)
            .unwrap()
            .canonical_value(),
    );
    let force_b = frame_b
        .internal_traction_world(&response_b)
        .unwrap()
        .canonical_value();
    for i in 0..3 {
        near(force_a[i], force_b[i]);
    }
    let ca = frame_a
        .tangent_world(&response_a)
        .unwrap()
        .canonical_value();
    let cb = frame_b
        .tangent_world(&response_b)
        .unwrap()
        .canonical_value();
    let map = [2, 0, 1];
    for i in 0..3 {
        for j in 0..3 {
            near(cb[i][j], ca[map[i]][map[j]]);
        }
    }
}
#[test]
fn shared_frame_normalization_has_no_length_unit_cutoff_or_overflow() {
    for scale in [1.0e-300, 1.0e-100, 1.0, 1.0e100, 1.0e300] {
        let unit =
            crate::physics::structural::frame::unit_vector([scale, 2.0 * scale, -scale]).unwrap();
        let expected = [1.0, 2.0, -1.0].map(|x| x / 6.0_f64.sqrt());
        for i in 0..3 {
            near(unit[i], expected[i]);
        }
    }
}
