use super::super::*;

#[test]
fn interior_star_quality_diagnostic_bins_candidate_quality() {
    let cavity = two_tetrahedron_bipyramid_cavity();
    let nodes = two_tetrahedron_bipyramid_nodes();
    let candidates = vec![
        ConstrainedCavityNode {
            node_id: 10,
            coordinates_m: [0.25, 0.25, 0.0],
        },
        ConstrainedCavityNode {
            node_id: 11,
            coordinates_m: [3.0, 3.0, 3.0],
        },
    ];

    let diagnostic = diagnostic_interior_star_quality(
        &cavity,
        &nodes,
        &candidates,
        ConstrainedCavityRefillOptions {
            min_scaled_jacobian: 0.01,
            volume_relative_tolerance: 1.0e-12,
            ..ConstrainedCavityRefillOptions::default()
        },
    )
    .expect("interior star diagnostic should evaluate");

    assert_eq!(diagnostic.candidate_count, 1);
    assert_eq!(diagnostic.pass_count, 1);
    assert!(diagnostic.max_min_scaled_jacobian >= 0.01);
    assert!(!diagnostic.min_scaled_jacobian_bins.is_empty());
    assert_eq!(
        diagnostic.rejected_by_reason,
        BTreeMap::from([("interior_point_outside_cavity", 1)])
    );
}
