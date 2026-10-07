use super::*;
#[test]
fn counted_index_rejects_bad_inventory_and_preserves_sparse_identity_keys() {
    for (values, count) in [
        (vec![(7u64, 1), (7, 2)], 2),
        (vec![(7, 1)], 2),
        (vec![(7, 1)], 0),
    ] {
        assert_eq!(
            SortedValues::derive(values.into_iter(), count).unwrap_err(),
            TopologyStorageError::Index
        );
    }
    let index = SortedValues::derive([(u64::MAX, -2), (0, 1), (17, 3)].into_iter(), 3).unwrap();
    assert_eq!(index.keys().copied().collect::<Vec<_>>(), [0, 17, u64::MAX]);
    assert_eq!(index.get(&u64::MAX), Some(&-2));
    assert_eq!(index.get(&18), None);
    assert_eq!(
        slots::<(u64, usize)>(usize::MAX),
        Err(TopologyStorageError::Capacity)
    );
}
