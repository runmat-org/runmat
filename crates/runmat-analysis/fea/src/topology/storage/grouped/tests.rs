use super::*;

#[test]
fn grouped_aliases_are_order_independent_and_preserve_real_ambiguity() {
    let pairs = [
        (u64::MAX, 8usize),
        (4, 2),
        (4, 1),
        (4, 2),
        (u64::MAX, 8),
        (0, 9),
    ];
    let expected = [(0, vec![9]), (4, vec![1, 2]), (u64::MAX, vec![8])];
    for values in [pairs.to_vec(), pairs.into_iter().rev().collect()] {
        let grouped = SortedValues::derive_grouped(values.into_iter(), pairs.len()).unwrap();
        assert_eq!(grouped.entries().cloned().collect::<Vec<_>>(), expected);
        assert_eq!(grouped.get(&5), None);
    }
    let empty = SortedValues::<u64, Vec<usize>>::derive_grouped(std::iter::empty(), 0).unwrap();
    assert_eq!(empty.len(), 0);
    for count in [0, 2] {
        assert_eq!(
            SortedValues::derive_grouped([(1u64, 3usize)].into_iter(), count).unwrap_err(),
            TopologyStorageError::Index
        );
    }
    assert_eq!(
        SortedValues::<u64, Vec<usize>>::grouped_construction_resident_bytes(usize::MAX),
        Err(TopologyStorageError::Capacity)
    );
}
