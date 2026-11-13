use super::*;
#[test]
fn packed_bounds_preserve_closed_candidates_under_permutations_and_linear_storage() {
    for count in [0, 1, 64, 4096] {
        let entries = (0..count)
            .map(|index| {
                let x = index as f64 * 2.0 - 20.0;
                SpatialEntry {
                    bounds: Aabb3 {
                        min_m: [x, -1.0, 0.0],
                        max_m: [x + 1.0, 1.0, 0.0],
                    },
                    payload: index,
                }
            })
            .collect::<Vec<_>>();
        for reverse in [false, true] {
            let mut original = entries.clone();
            if reverse {
                original.reverse();
            }
            let profile = PackedBoundsSpatialIndex::<usize>::storage_profile(count).unwrap();
            let index =
                PackedBoundsSpatialIndex::from_entries_with_checkpoint(original, &mut || {
                    Ok::<_, ()>(())
                })
                .unwrap()
                .unwrap();
            assert_eq!(
                profile.retained_bytes,
                (index.nodes.capacity() * std::mem::size_of::<Node<usize>>()) as u64
            );
            for x in [-20.0, -19.0, -18.5, 0.0, 3000.0, 8192.0] {
                let bounds = Aabb3 {
                    min_m: [x, -1.0, 0.0],
                    max_m: [x + 0.5, 1.0, 0.0],
                };
                let expected = entries
                    .iter()
                    .filter(|entry| entry.bounds.intersects(bounds))
                    .map(|entry| entry.payload)
                    .collect::<Vec<_>>();
                let mut actual = Vec::new();
                let mut probes = 0;
                index
                    .visit_bounds_with_checkpoint(
                        bounds,
                        &mut |entry| actual.push(entry.payload),
                        &mut || {
                            probes += 1;
                            Ok::<_, ()>(())
                        },
                    )
                    .unwrap();
                actual.sort_unstable();
                assert_eq!(actual, expected);
                assert!(
                    probes <= 65,
                    "separated query visited {probes} subtrees for {count} entries"
                );
            }
        }
    }
    assert!(PackedBoundsSpatialIndex::<usize>::storage_profile(usize::MAX).is_none());
}
#[test]
fn packed_build_and_query_preserve_control_failures_and_reject_nonfinite_bounds() {
    let entry = SpatialEntry {
        bounds: Aabb3 {
            min_m: [0.0; 3],
            max_m: [1.0; 3],
        },
        payload: 1usize,
    };
    let mut total = 0;
    let index = PackedBoundsSpatialIndex::from_entries_with_checkpoint(
        vec![entry.clone(); 32],
        &mut || {
            total += 1;
            Ok::<_, usize>(())
        },
    )
    .unwrap()
    .unwrap();
    for stop in 1..=total {
        let mut calls = 0;
        assert!(
            matches!(PackedBoundsSpatialIndex::from_entries_with_checkpoint(vec![entry.clone();32], &mut || { calls+=1; if calls==stop { Err(stop) } else { Ok(()) } }), Err(value) if value==stop)
        );
    }
    let mut bad = entry.clone();
    bad.bounds.min_m[0] = f64::NAN;
    assert!(PackedBoundsSpatialIndex::from_entries_with_checkpoint(
        vec![bad],
        &mut || Ok::<_, ()>(())
    )
    .unwrap()
    .is_none());
    let result = index.visit_bounds_with_checkpoint(entry.bounds, &mut |_| {}, &mut || Err(91));
    assert_eq!(result, Err(91));
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Streaming packed queries stop at the first caller predicate failure without a second
//   candidate buffer; original errors remain observable.
#[test]
fn fallible_packed_visitor_preserves_first_predicate_failure_without_buffering() {
    let bounds = Aabb3 {
        min_m: [0.0; 3],
        max_m: [1.0; 3],
    };
    let index = PackedBoundsSpatialIndex::from_entries_with_checkpoint(
        (0..128)
            .map(|payload| SpatialEntry { bounds, payload })
            .collect(),
        &mut || Ok::<_, &str>(()),
    )
    .unwrap()
    .unwrap();
    let mut visited = 0;
    let error = index
        .try_visit_bounds_with_checkpoint(
            bounds,
            &mut |_| {
                visited += 1;
                Err("original predicate failure")
            },
            &mut || Ok(()),
        )
        .unwrap_err();
    assert_eq!(visited, 1);
    assert_eq!(error, "original predicate failure");
}
