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
        unique_nodes(parents.into_iter().flatten(), 4, limits(4)).unwrap(),
        [0, 1, 2, 3]
    );
    assert!(unique_nodes(parents.into_iter().flatten(), 4, limits(3)).is_err());
    assert_eq!(
        unique_nodes(std::iter::repeat_n(7, 1000), 8, limits(1)).unwrap(),
        [7]
    );
    // Source traversal order must never leak into canonical target identity.
    assert_eq!(
        unique_nodes((0..4096).rev().chain(0..4096), 4096, limits(4096)).unwrap(),
        (0..4096).collect::<Vec<_>>()
    );
}

#[test]
fn union_stops_before_consuming_beyond_first_excess_unique_node() {
    let mut consumed = 0;
    let result = unique_nodes(
        [3, 3, 1, 3, 5, 99].into_iter().inspect(|_| consumed += 1),
        100,
        limits(2),
    );
    assert!(result.is_err());
    assert_eq!(consumed, 5);
}

#[test]
fn known_reservation_overflow_is_fallible() {
    assert!(reserved::<StableDigest>(usize::MAX).is_err());
}

fn limits(maximum: usize) -> ObservationLimits {
    ObservationLimits {
        maximum_selected_entities: maximum,
        ..Default::default()
    }
}

#[test]
fn nodal_selection_admits_complete_union_and_identity_peak_before_consumption() {
    for domain in [4, 10, 65, 4096] {
        let bytes = node_union_resident_bytes(domain, domain).unwrap();
        let mut consumed = 0;
        let limits = ObservationLimits {
            maximum_selected_entities: domain,
            maximum_peak_resident_bytes: bytes - 1,
            ..Default::default()
        };
        let (short, retained, peak) = crate::topology::measure_test_allocations(|| {
            unique_nodes((0..domain).inspect(|_| consumed += 1), domain, limits)
        });
        assert!(matches!(
            short,
            Err(ObservationError::Limit(
                "target selection construction residency"
            ))
        ));
        assert_eq!((consumed, retained, peak), (0, 0, 0));
        let limits = ObservationLimits {
            maximum_peak_resident_bytes: bytes,
            ..limits
        };
        let (result, _, peak) = crate::topology::measure_test_allocations(|| {
            unique_nodes((0..domain).rev().chain(0..domain), domain, limits)
        });
        assert_eq!(result.unwrap(), (0..domain).collect::<Vec<_>>());
        assert!(peak as u64 <= bytes);
    }
}
