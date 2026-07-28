use super::*;
use crate::quality::tolerance::MeshingTolerance;

#[test]
fn tetrahedron_signed_volume_reports_orientation() {
    let tetrahedron = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    assert!((tetrahedron_signed_volume(tetrahedron) - 1.0 / 6.0).abs() < 1.0e-12);
    let (node_ids, volume) = orient_tetrahedron_node_ids(
        [1, 3, 2, 4],
        [
            tetrahedron[0],
            tetrahedron[2],
            tetrahedron[1],
            tetrahedron[3],
        ],
    );
    assert_eq!(node_ids, [1, 2, 3, 4]);
    assert!((volume - 1.0 / 6.0).abs() < 1.0e-12);
}

#[test]
fn tetrahedron_scaled_jacobian_reports_shape_quality() {
    let orthogonal = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let sliver = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0e-4, 1.0e-4, 1.0e-5],
    ];

    assert!(tetrahedron_scaled_jacobian(orthogonal) > 0.5);
    assert!(tetrahedron_scaled_jacobian(sliver) < 0.01);
}

#[test]
fn ray_triangle_intersection_returns_distance_and_barycentric_coordinates() {
    let hit = ray_triangle_intersection(
        [-1.0, 0.25, 0.25],
        [1.0, 0.0, 0.0],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        MeshingTolerance::default(),
    )
    .expect("ray should hit triangle");

    assert!((hit.distance - 1.0).abs() < 1.0e-12);
    assert!((hit.barycentric_u - 0.25).abs() < 1.0e-12);
    assert!((hit.barycentric_v - 0.25).abs() < 1.0e-12);
}

#[test]
fn point_in_closed_surface_classifies_cube_points() {
    let triangles = cube_triangles();
    let tolerance = MeshingTolerance::from_bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

    assert_eq!(
        point_in_closed_triangle_surface([0.5, 0.5, 0.5], &triangles, tolerance),
        PointInClosedSurface::Inside
    );
    assert_eq!(
        point_in_closed_triangle_surface([1.5, 0.5, 0.5], &triangles, tolerance),
        PointInClosedSurface::Outside
    );
    assert_eq!(
        point_in_closed_triangle_surface([1.0, 0.5, 0.5], &triangles, tolerance),
        PointInClosedSurface::OnBoundary
    );
}

#[test]
fn circumsphere_contains_regular_tetrahedron_center() {
    let tetrahedron = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    assert!(tetrahedron_circumsphere_contains_point(
        tetrahedron,
        [0.5, 0.5, 0.5],
        MeshingTolerance::default()
    ));
    assert!(!tetrahedron_circumsphere_contains_point(
        tetrahedron,
        [2.0, 2.0, 2.0],
        MeshingTolerance::default()
    ));
}

fn cube_triangles() -> Vec<Triangle3> {
    let p = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    [
        [p[0], p[2], p[1]],
        [p[0], p[3], p[2]],
        [p[4], p[5], p[6]],
        [p[4], p[6], p[7]],
        [p[0], p[1], p[5]],
        [p[0], p[5], p[4]],
        [p[1], p[2], p[6]],
        [p[1], p[6], p[5]],
        [p[2], p[3], p[7]],
        [p[2], p[7], p[6]],
        [p[3], p[0], p[4]],
        [p[3], p[4], p[7]],
    ]
    .to_vec()
}
