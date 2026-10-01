use super::*;

#[test]
fn exact_known_selection_count_is_inclusive() {
    assert_eq!(selected_indices(0..4, 4).unwrap(), [0, 1, 2, 3]);
    assert!(matches!(
        selected_indices(0..4, 3),
        Err(ObservationError::Limit("selected topology entities"))
    ));
    assert!(selected_indices(std::iter::once(0), 0).is_err());
    assert!(selected_indices(0..0, 0).unwrap().is_empty());
}

#[test]
fn shared_parent_nodes_count_unique_union_not_connectivity() {
    let parents = [[3, 0, 1], [1, 3, 2], [0, 2, 1], [3, 1, 0]];
    assert_eq!(
        unique_nodes(parents.into_iter().flatten(), 4).unwrap(),
        [0, 1, 2, 3]
    );
    assert!(unique_nodes(parents.into_iter().flatten(), 3).is_err());
    assert_eq!(unique_nodes(std::iter::repeat_n(7, 1000), 1).unwrap(), [7]);
    // Hash iteration order must never leak into canonical target identity.
    assert_eq!(
        unique_nodes((0..4096).rev().chain(0..4096), 4096).unwrap(),
        (0..4096).collect::<Vec<_>>()
    );
}

#[test]
fn union_stops_before_consuming_beyond_first_excess_unique_node() {
    let mut consumed = 0;
    let result = unique_nodes(
        [3, 3, 1, 3, 5, 99].into_iter().inspect(|_| consumed += 1),
        2,
    );
    assert!(result.is_err());
    assert_eq!(consumed, 5);
}

#[test]
fn known_reservation_overflow_is_fallible() {
    assert!(reserved::<StableDigest>(usize::MAX).is_err());
}
