use super::*;
use crate::cdt::{build_delaunay_volume_point_set, DelaunayPointSetOptions};

#[test]
fn one_facet_recovers_both_sides_without_unnecessary_insertions() {
    let constraints = constraints();
    let point_set = build_delaunay_volume_point_set(
        constraints.volume_nodes(),
        DelaunayPointSetOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let segments = recover_delaunay_segments(
        point_set,
        &constraints,
        DelaunaySegmentRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let recovered = recover_delaunay_facets(
        segments.clone(),
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let repeated = recover_delaunay_facets(
        segments.clone(),
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    assert_eq!(recovered, repeated);
    assert!(recovered.steiner_insertions.is_empty());
    validate_delaunay_facet_recovery(
        &recovered,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    let bounded = recover_delaunay_facets(
        segments,
        &constraints,
        DelaunayFacetRecoveryOptions {
            maximum_cavity_steiner_nodes: 1,
            ..DelaunayFacetRecoveryOptions::default()
        },
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(bounded, recovered);
}

fn constraints() -> DelaunayConstraints {
    crate::cdt::test_fixtures::same_facet_multi_steiner_constraints()
}
