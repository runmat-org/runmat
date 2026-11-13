use super::*;
fn replace(owner: &mut TopologyMutation) {
    owner
        .replace_cavity(
            &BTreeSet::from([0, 1]),
            vec![
                ([0, 1, 2, 3], None),
                ([0, 1, 3, 4], None),
                ([0, 1, 4, 2], None),
            ],
        )
        .unwrap();
}
#[test]
fn nested_queries_restore_preexisting_pending_updates_after_cancellation_and_commit() {
    let original = mesh(4);
    let original_candidates = query(&original, bounds());
    let mut probe = original.clone();
    replace(&mut probe);
    let mut total = 0;
    probe
        .prepare_cell_spatial_index(&mut || {
            total += 1;
            Ok::<_, ()>(())
        })
        .unwrap();
    assert!(total > 2);
    let expected_candidates = query(&probe, bounds());
    for cancel_at in 1..=total {
        let mut owner = original.clone();
        {
            let mut parent = owner.cavity_trial();
            replace(&mut parent);
            let pending = parent
                .cell_spatial_index
                .borrow()
                .as_ref()
                .unwrap()
                .pending
                .len();
            let original_undo = parent
                .cell_spatial_index
                .borrow()
                .as_ref()
                .unwrap()
                .pending
                .undo_entries();
            assert!(pending > 0);
            {
                let child = parent.cavity_trial();
                assert_eq!(
                    child
                        .cell_spatial_index
                        .borrow()
                        .as_ref()
                        .unwrap()
                        .pending
                        .undo_entries(),
                    original_undo
                );
                let mut calls = 0;
                assert!(
                    child
                        .prepare_cell_spatial_index(&mut || {
                            calls += 1;
                            if calls == cancel_at {
                                Err("cancel")
                            } else {
                                Ok(())
                            }
                        })
                        .is_err(),
                    "cancellation checkpoint {cancel_at}/{total}"
                );
                assert_eq!(query(&child, bounds()), expected_candidates);
            }
            assert_eq!(
                parent
                    .cell_spatial_index
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .pending
                    .len(),
                pending
            );
            {
                let child = parent.cavity_trial();
                assert_eq!(query(&child, bounds()), expected_candidates);
                child.commit();
            }
            assert!(parent
                .cell_spatial_index
                .borrow()
                .as_ref()
                .unwrap()
                .pending
                .is_empty());
        }
        assert_eq!(query(&owner, bounds()), original_candidates);
        assert!(owner
            .cell_spatial_index
            .borrow()
            .as_ref()
            .unwrap()
            .pending
            .is_empty());
    }
}
