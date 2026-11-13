use super::{boundary_shape, visit_boundary_quadrature, weighted_derivative, DUNAVANT_SEVEN};
use crate::topology::element::{cross, dot};
use runmat_meshing_core::BoundaryTriangleOrder;

#[test]
fn shared_face_rule_rejects_a_fold_missed_by_dunavant_points() {
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
    // The analytic mapping has signed area q(u,v)=2.04-2.08u-1.04v; the shared rule finds
    // its exact negative minimum, and boundary integration refuses the face.
    let validity =
        runmat_meshing_core::boundary_face_validity(BoundaryTriangleOrder::Tri6, &coordinates)
            .unwrap();
    assert!((validity.minimum_normal_jacobian + 0.04).abs() < 1.0e-14);
    assert!(!validity.is_valid());
    assert!(visit_boundary_quadrature(BoundaryTriangleOrder::Tri6, &coordinates, |_| {}).is_none());
}
