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

    assert!(tetrahedron_scaled_jacobian(orthogonal).unwrap() > 0.5);
    assert!(tetrahedron_scaled_jacobian(sliver).unwrap() < 0.01);
}

#[test]
fn scaled_jacobian_reserves_reference_work_and_settles_invalid_prefix() {
    use runmat_simulation_usage::{OperationCount, UsageContext, UsageLedger};
    use std::sync::Arc;

    let valid = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    // Each call reserves its full batch before settling the work it completes.
    let budget = OperationCount::from_operations(280 + 280);
    let ledger = Arc::new(UsageLedger::new(budget, budget).unwrap());
    let context = UsageContext::new(Arc::clone(&ledger));
    context.run_sync(|| {
        tetrahedron_scaled_jacobian(valid).unwrap();
        assert_eq!(tetrahedron_scaled_jacobian([[0.0; 3]; 4]).unwrap(), 0.0);
    });
    let snapshot = ledger.snapshot().unwrap();
    assert_eq!(
        snapshot.completed.meshing,
        OperationCount::from_operations(280 + 12)
    );
    assert_eq!(snapshot.reserved, OperationCount::ZERO);
}

#[test]
fn tetrahedron_scaled_jacobian_is_invariant_under_small_uniform_scales() {
    let tetrahedron = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let small = tetrahedron.map(|point| point.map(|coordinate| coordinate * 1.0e-9));

    let reference = tetrahedron_scaled_jacobian(tetrahedron).unwrap();
    let scaled = tetrahedron_scaled_jacobian(small).unwrap();

    assert!(reference > 0.0);
    assert!((scaled - reference).abs() <= 32.0 * f64::EPSILON);
}

#[test]
fn tetrahedron_scaled_jacobian_retains_an_extreme_positive_sliver() {
    let sliver = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0e-200],
    ];

    let quality = tetrahedron_scaled_jacobian(sliver).unwrap();
    assert!(quality > 0.0);
    assert!(quality < 1.0e-190);
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

#[test]
fn circumsphere_radius_is_translation_invariant() {
    let tetrahedron = [
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [0.0, 0.25, 0.0],
        [0.0, 0.0, 0.25],
    ];
    let translated = tetrahedron.map(|point| point.map(|coordinate| coordinate + 1.0e12));
    let tolerance = MeshingTolerance {
        absolute_m: runmat_types::physical::Length::new(0.0),
        relative: 0.0,
    };
    let (_, original_radius_squared) = tetrahedron_circumsphere(tetrahedron, tolerance).unwrap();
    let (_, translated_radius_squared) = tetrahedron_circumsphere(translated, tolerance).unwrap();

    assert_eq!(original_radius_squared, translated_radius_squared);
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

/// A 0.137 mm by 2.4 um needle on the docs heat sink's fin root (q-mesh-regress): a point inside
/// it projects onto it. The degeneracy test is relative to the edge lengths, so small triangles
/// in metres are not treated as collinear.
#[test]
fn closest_point_projects_onto_a_small_needle_triangle() {
    let triangle = [
        [0.06224886, 0.00246126, 0.02583984375],
        [0.06224886, 0.00246126, 0.025703125],
        [0.062246427569715664, 0.0024612718337142497, 0.025768943772598276],
    ];
    let point = [0.06224806975105429, 0.002461263844541926, 0.02576892547318599];
    let closest = closest_point_on_triangle(point, triangle);
    let gap = (0..3)
        .map(|axis| (closest[axis] - point[axis]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(gap < 1.0e-15, "{gap}");
    // A collinear triangle gives the closest point of its segment.
    let collinear = [[0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0], [2.0e-4, 0.0, 0.0]];
    let on_segment = closest_point_on_triangle([1.9e-4, 1.0e-5, 0.0], collinear);
    assert!((on_segment[0] - 1.9e-4).abs() < 1.0e-18 && on_segment[1] == 0.0);
}
