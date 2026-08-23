use super::*;
use crate::cdt::{build_delaunay_volume_point_set, DelaunayPointSetOptions};

#[test]
fn one_facet_recovers_both_unfillable_sides_with_bounded_insertions() {
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
    assert_eq!(recovered.steiner_insertions.len(), 2);
    assert!(recovered
        .steiner_insertions
        .iter()
        .all(|insertion| insertion.constraint_index == 0));
    assert_eq!(
        recovered
            .steiner_insertions
            .iter()
            .map(|insertion| insertion.insertion_round)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    validate_delaunay_facet_recovery(
        &recovered,
        &constraints,
        DelaunayFacetRecoveryOptions::default(),
        &NeverCancelled,
    )
    .unwrap();

    let failure = recover_delaunay_facets(
        segments,
        &constraints,
        DelaunayFacetRecoveryOptions {
            maximum_cavity_steiner_nodes: 1,
            ..DelaunayFacetRecoveryOptions::default()
        },
        &NeverCancelled,
    )
    .unwrap_err();
    assert_eq!(failure.kind, DelaunayFacetRecoveryErrorKind::ResourceLimit);
}

fn constraints() -> DelaunayConstraints {
    crate::cdt::test_fixtures::same_facet_multi_steiner_constraints()
}
