use super::*;

#[test]
fn dense_union_matches_independent_set_across_word_boundaries() {
    for domain in [0, 1, 63, 64, 65, 129, 4096] {
        let source = (0..domain)
            .rev()
            .chain(0..domain)
            .filter(|i| i % 3 != 1)
            .collect::<Vec<_>>();
        let expected = source
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(
            DenseIndexUnion::new(domain, domain)
                .unwrap()
                .collect(source.into_iter())
                .unwrap(),
            expected
        );
    }
}

#[test]
fn dense_union_has_counted_requested_storage_and_early_unique_limit() {
    for domain in [0, 1, 64, 65, 4096] {
        let bytes = DenseIndexUnion::resident_bytes(domain, domain).unwrap();
        let (result, _, peak) = crate::topology::measure_test_allocations(|| {
            DenseIndexUnion::new(domain, domain)
                .unwrap()
                .collect((0..domain).rev())
        });
        assert_eq!(result.unwrap().len(), domain);
        assert_eq!(peak as u64, bytes);
    }
    let mut consumed = 0;
    let result = DenseIndexUnion::new(100, 2)
        .unwrap()
        .collect([3, 3, 1, 3, 5, 99].into_iter().inspect(|_| consumed += 1));
    assert_eq!(result, Err(DenseUnionError::Limit));
    assert_eq!(consumed, 5);
    assert_eq!(
        DenseIndexUnion::new(3, 3).unwrap().collect([3].into_iter()),
        Err(DenseUnionError::Domain)
    );
    assert_eq!(
        DenseIndexUnion::resident_bytes(usize::MAX, usize::MAX),
        Err(TopologyStorageError::Capacity)
    );
}

#[test]
fn cancelled_dense_union_preserves_typed_cancellation() {
    let union = DenseIndexUnion::new(4, 4).unwrap();
    let token = crate::FeaCancellationToken::new();
    token.cancel();
    let _guard = crate::replace_fea_progress_context(crate::FeaProgressContext {
        cancellation: token,
        ..Default::default()
    });
    assert_eq!(
        union.collect((0..4).rev()),
        Err(DenseUnionError::Storage(TopologyStorageError::Cancelled))
    );
    assert!(matches!(
        DenseIndexUnion::new(4, 4),
        Err(TopologyStorageError::Cancelled)
    ));
}
