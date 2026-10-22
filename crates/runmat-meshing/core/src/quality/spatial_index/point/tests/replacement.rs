use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct Counted {
    value: usize,
    copies: Arc<AtomicUsize>,
}
impl Clone for Counted {
    fn clone(&self) -> Self {
        self.copies.fetch_add(1, Ordering::Relaxed);
        Self {
            value: self.value,
            copies: self.copies.clone(),
        }
    }
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Interior replacement in 1,024- and 65,536-point inventories copies one 32-point
//   payload page while the original index stays intact. This excludes relocation outside retained
//   routing bounds, complete mesh mutation, measured whole-task demand and arbitrary native model
//   scale.
#[test]
fn interior_point_replacement_copies_one_page_and_preserves_the_original_index() {
    let mut work = Vec::new();
    for count in [1024, 65536] {
        let copies = Arc::new(AtomicUsize::new(0));
        let index = PointSpatialIndex::from_entries_with_checkpoint(
            (0..count)
                .map(|value| {
                    SpatialPoint::new(
                        [value as f64, 0.0, 0.0],
                        Counted {
                            value,
                            copies: copies.clone(),
                        },
                    )
                    .unwrap()
                })
                .collect(),
            &mut || Ok::<_, ()>(()),
        )
        .unwrap();
        let mut changed = index
            .clone_with_checkpoint(&mut || Ok::<_, ()>(()))
            .unwrap();
        let mut checkpoints = 0;
        assert!(changed
            .replace_with_checkpoint(
                count / 2,
                SpatialPoint::new(
                    [0.25, 0.0, 0.0],
                    Counted {
                        value: count,
                        copies: copies.clone()
                    }
                )
                .unwrap(),
                &mut || {
                    checkpoints += 1;
                    Ok::<_, ()>(())
                }
            )
            .unwrap());
        assert_eq!(copies.load(Ordering::Relaxed), 32);
        assert_eq!(index.len(), changed.len());
        for (slot, (original, replacement)) in index.iter().zip(changed.iter()).enumerate() {
            assert_eq!(original.payload.value, slot);
            if slot == count / 2 {
                assert_eq!(replacement.position(), [0.25, 0.0, 0.0]);
            } else {
                assert_eq!(original.position(), replacement.position());
            }
        }
        let found = changed
            .query_bounds_with_checkpoint(
                Aabb3 {
                    min_m: [0.25, 0.0, 0.0],
                    max_m: [0.25, 0.0, 0.0],
                },
                &mut || Ok::<_, ()>(()),
            )
            .unwrap();
        assert_eq!(
            found
                .iter()
                .map(|entry| entry.payload.value)
                .collect::<Vec<_>>(),
            vec![count]
        );
        let removed = changed
            .query_bounds_with_checkpoint(
                Aabb3 {
                    min_m: [(count / 2) as f64, 0.0, 0.0],
                    max_m: [(count / 2) as f64, 0.0, 0.0],
                },
                &mut || Ok::<_, ()>(()),
            )
            .unwrap();
        assert!(removed.is_empty());
        work.push(checkpoints);
    }
    assert!(work[1] <= work[0] * 4, "local routing growth: {work:?}");
}

#[test]
fn replacement_queries_match_exhaustive_construction_bounds() {
    let mut entries = points();
    let mut index =
        PointSpatialIndex::from_entries_with_checkpoint(entries.clone(), &mut || Ok::<_, ()>(()))
            .unwrap();
    for slot in [0, 37, 7777] {
        let position = [0.36, 0.005, 0.004];
        let bounds = Aabb3 {
            min_m: [0.359, 0.004, 0.003],
            max_m: [0.361, 0.006, 0.005],
        };
        entries[slot] = SpatialPoint::with_bounds(position, bounds, slot).unwrap();
        assert!(index
            .replace_with_checkpoint(slot, entries[slot].clone(), &mut || Ok::<_, ()>(()))
            .unwrap());
    }
    for x in [0.0, 0.359, 0.36, 0.361, 0.88] {
        let bounds = Aabb3 {
            min_m: [x, 0.0, 0.0],
            max_m: [x, 0.01, 0.012],
        };
        let expected = entries
            .iter()
            .filter(|entry| entry.bounds().intersects(bounds))
            .map(|entry| entry.payload)
            .collect::<Vec<_>>();
        assert_eq!(query(&index, bounds), expected);
    }
}

#[test]
fn cancellation_at_every_replacement_checkpoint_leaves_the_index_unchanged() {
    let entries = (0..256)
        .map(|value| SpatialPoint::new([value as f64; 3], value).unwrap())
        .collect();
    let original =
        PointSpatialIndex::from_entries_with_checkpoint(entries, &mut || Ok::<_, ()>(())).unwrap();
    let replacement = SpatialPoint::new([0.25; 3], 1000).unwrap();
    let mut probe = original.clone();
    let mut steps = 0;
    probe
        .replace_with_checkpoint(128, replacement.clone(), &mut || {
            steps += 1;
            Ok::<_, ()>(())
        })
        .unwrap();
    for budget in 0..steps {
        let mut trial = original.clone();
        let mut visits = 0;
        assert!(trial
            .replace_with_checkpoint(128, replacement.clone(), &mut || {
                if visits == budget {
                    Err(())
                } else {
                    visits += 1;
                    Ok(())
                }
            })
            .is_err());
        assert!(Arc::ptr_eq(
            trial.entries.root.as_ref().unwrap(),
            original.entries.root.as_ref().unwrap()
        ));
        assert!(Arc::ptr_eq(
            trial.root.as_ref().unwrap(),
            original.root.as_ref().unwrap()
        ));
    }
    assert!(!probe
        .replace_with_checkpoint(256, replacement, &mut || Ok::<_, ()>(()))
        .unwrap());
}
