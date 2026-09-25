use super::{
    boundary_shape, quadratic_edge_minimum, signed_boundary_area_minimum, weighted_derivative,
    DUNAVANT_SEVEN,
};
use crate::topology::element::{cross, dot};
use runmat_meshing_core::BoundaryTriangleOrder;

#[test]
fn edge_quadratic_finds_a_negative_minimum_between_positive_endpoints() {
    // q(t) = (t - 0.3)^2 - 0.01 is positive at both endpoints and its midpoint.
    assert!((quadratic_edge_minimum(0.08, 0.48, 0.03) + 0.01).abs() < 1.0e-14);
}

#[test]
fn exact_tri6_orientation_check_finds_a_fold_missed_by_dunavant_points() {
    let coordinates = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.76, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
    ];
    let corner_normal = [0.0, 0.0, 1.0];
    for (barycentric, _) in DUNAVANT_SEVEN {
        let (_, derivatives) = boundary_shape(BoundaryTriangleOrder::Tri6, barycentric);
        let tangent_u = weighted_derivative(&coordinates, &derivatives, 0);
        let tangent_v = weighted_derivative(&coordinates, &derivatives, 1);
        assert!(dot(cross(tangent_u, tangent_v), corner_normal) > 0.0);
    }
    let minimum = signed_boundary_area_minimum(&coordinates, corner_normal).unwrap();
    // The analytic mapping has signed area q(u,v)=2.04-2.08u-1.04v.
    assert!((minimum + 0.04).abs() < 1.0e-14);
}
