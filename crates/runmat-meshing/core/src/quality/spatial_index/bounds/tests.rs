use super::*;

fn entry(id: u32) -> SpatialEntry<u32> {
    let x = id as f64 * 3.0;
    SpatialEntry {
        bounds: Aabb3 {
            min_m: [x, -1.0, 0.0],
            max_m: [x + 1.0, 1.0, 2.0],
        },
        payload: id,
    }
}
fn query(index: &BoundsSpatialIndex<u32>, bounds: Aabb3) -> Vec<u32> {
    let mut result = Vec::new();
    index
        .visit_bounds_with_checkpoint(bounds, &mut |entry| result.push(entry.payload), &mut || {
            Ok::<_, ()>(())
        })
        .unwrap();
    result.sort_unstable();
    result
}
fn all() -> Aabb3 {
    Aabb3 {
        min_m: [-1.0e8; 3],
        max_m: [1.0e8; 3],
    }
}

#[test]
fn bounds_candidates_match_exhaustive_closed_box_queries_after_updates() {
    let mut entries = (0..512).map(entry).collect::<Vec<_>>();
    let mut index =
        BoundsSpatialIndex::from_entries_with_checkpoint(entries.clone(), &mut || Ok::<_, ()>(()))
            .unwrap()
            .unwrap();
    for id in (0..512).step_by(3) {
        assert!(index
            .remove_with_checkpoint(&entry(id), &mut || Ok::<_, ()>(()))
            .unwrap());
        entries.retain(|entry| entry.payload != id);
    }
    for id in 512..1024 {
        let next = entry(id);
        assert!(index
            .insert_with_checkpoint(next.clone(), &mut || Ok::<_, ()>(()))
            .unwrap());
        entries.push(next);
    }
    assert_eq!(
        index.root.as_ref().unwrap().assert_balanced(),
        entries.len()
    );
    for id in (0..1024).step_by(7) {
        let mut bounds = entry(id).bounds;
        bounds.min_m[0] -= 1.0;
        bounds.max_m[0] += 3.0;
        let mut expected = entries
            .iter()
            .filter(|entry| entry.bounds.intersects(bounds))
            .map(|entry| entry.payload)
            .collect::<Vec<_>>();
        expected.sort_unstable();
        assert_eq!(query(&index, bounds), expected);
    }
    // Contact on a face, edge or point must never be pruned.
    assert_eq!(
        query(
            &index,
            Aabb3 {
                min_m: [4.0, 1.0, 2.0],
                max_m: [4.0, 1.0, 2.0]
            }
        ),
        vec![1]
    );
}

#[test]
fn balanced_query_work_and_persistent_trial_storage_are_bounded() {
    let mut costs = Vec::new();
    for count in [64, 4096] {
        let index =
            BoundsSpatialIndex::from_entries_with_checkpoint((0..count).map(entry), &mut || {
                Ok::<_, ()>(())
            })
            .unwrap()
            .unwrap();
        let mut trial = index.clone();
        assert!(Arc::ptr_eq(
            index.root.as_ref().unwrap(),
            trial.root.as_ref().unwrap()
        ));
        let mut work = 0;
        let mut found = Vec::new();
        trial
            .visit_bounds_with_checkpoint(
                entry(count / 2).bounds,
                &mut |entry| found.push(entry.payload),
                &mut || {
                    work += 1;
                    Ok::<_, ()>(())
                },
            )
            .unwrap();
        assert_eq!(found, vec![count / 2]);
        costs.push(work);
        assert!(trial
            .insert_with_checkpoint(entry(count + 10), &mut || Ok::<_, ()>(()))
            .unwrap());
        assert_eq!(query(&index, entry(count + 10).bounds), Vec::<u32>::new());
        assert_eq!(query(&trial, entry(count + 10).bounds), vec![count + 10]);
    }
    assert!(
        costs[1] <= costs[0] * 4,
        "64/4096 box query work: {costs:?}"
    );
}

#[test]
fn rejected_build_and_mutation_leave_no_partial_inventory() {
    let mut index =
        BoundsSpatialIndex::from_entries_with_checkpoint((0..64).map(entry), &mut || {
            Ok::<_, ()>(())
        })
        .unwrap()
        .unwrap();
    let expected = query(&index, all());
    for limit in 1..8 {
        let root = index.root.clone();
        let mut calls = 0;
        let result = index.insert_with_checkpoint(entry(100), &mut || {
            calls += 1;
            if calls == limit {
                Err("cancelled")
            } else {
                Ok(())
            }
        });
        assert_eq!(result, Err("cancelled"));
        assert!(Arc::ptr_eq(
            root.as_ref().unwrap(),
            index.root.as_ref().unwrap()
        ));
        assert_eq!(query(&index, all()), expected);
    }
    let mut calls = 0;
    assert!(
        BoundsSpatialIndex::from_entries_with_checkpoint((0..64).map(entry), &mut || {
            calls += 1;
            if calls == 10 {
                Err("budget")
            } else {
                Ok(())
            }
        })
        .is_err()
    );
    let invalid = SpatialEntry {
        bounds: Aabb3 {
            min_m: [f64::NAN; 3],
            max_m: [1.0; 3],
        },
        payload: 1,
    };
    assert!(!index
        .insert_with_checkpoint(invalid, &mut || Ok::<_, ()>(()))
        .unwrap());
    assert_eq!(query(&index, all()), expected);
}
