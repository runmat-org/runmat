use super::*;

#[test]
fn changed_edge_candidates_are_local_and_nested_rollback_coalesces() {
    let expected = BTreeSet::from([
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 3],
        [2, 4],
    ]);
    let mut counts = Vec::new();
    for count in [1, 64] {
        let mut topology = neighborhoods(count);
        let original = topology.clone();
        {
            let mut transaction = topology.cavity_trial();
            let mut outer = transaction.observe_edge_changes();
            {
                let mut inner = outer.cavity_trial();
                flip(&mut inner);
            }
            assert!(outer
                .removed_edge_candidates(&mut || Ok::<_, ()>(()))
                .unwrap()
                .is_empty());
            {
                let mut inner = outer.cavity_trial();
                flip(&mut inner);
                inner.commit();
            }
            let mut checks = 0;
            let candidates = outer
                .removed_edge_candidates(&mut || {
                    checks += 1;
                    Ok::<_, ()>(())
                })
                .unwrap();
            assert_eq!(candidates, expected);
            counts.push(checks);
            assert_eq!(
                outer.removed_edge_candidates(&mut || Err::<(), _>("cancelled")),
                Err("cancelled")
            );
        }
        assert_storage(&topology, &original);
    }
    assert_eq!(
        counts[0], counts[1],
        "unrelated neighborhoods add no notification work"
    );
}
