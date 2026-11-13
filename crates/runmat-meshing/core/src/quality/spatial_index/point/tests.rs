use super::*;
mod pages;
mod replacement;
fn points() -> Vec<SpatialPoint<usize>> {
    (0..12000)
        .map(|index| {
            let fin = index % 12;
            let position = [
                fin as f64 * 0.08,
                (index % 101) as f64 * 0.0001,
                (index / 101) as f64 * 0.0001,
            ];
            SpatialPoint::new(position, index).unwrap()
        })
        .collect()
}
fn query(index: &PointSpatialIndex<usize>, bounds: Aabb3) -> Vec<usize> {
    let ordered = index
        .query_bounds_with_checkpoint(bounds, &mut || Ok::<_, ()>(()))
        .unwrap()
        .into_iter()
        .map(|entry| entry.payload)
        .collect::<Vec<_>>();
    let mut streamed = Vec::new();
    index
        .visit_bounds_with_checkpoint(
            bounds,
            &mut |entry| streamed.push(entry.payload),
            &mut || Ok::<_, ()>(()),
        )
        .unwrap();
    streamed.sort_unstable();
    assert_eq!(streamed, ordered);
    ordered
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Twelve clustered thin-fin point inventories compare adaptive candidates with
//   exhaustive closed-boundary queries in original inventory order. Exact meshing predicates remain
//   outside this component; no original heat acceptance or whole-task scalability claim.
#[test]
fn clustered_fins_match_exhaustive_closed_boundary_queries() {
    let entries = points();
    let index =
        PointSpatialIndex::from_entries_with_checkpoint(entries.clone(), &mut || Ok::<_, ()>(()))
            .unwrap();
    for fin in 0..12 {
        let bounds = Aabb3 {
            min_m: [fin as f64 * 0.08, 0.002, 0.003],
            max_m: [fin as f64 * 0.08 + 0.0001, 0.003, 0.005],
        };
        let expected = entries
            .iter()
            .filter(|entry| bounds.contains_point(entry.position()))
            .map(|entry| entry.payload)
            .collect::<Vec<_>>();
        assert_eq!(query(&index, bounds), expected);
    }
}
#[test]
fn sparse_cluster_queries_do_not_scan_the_point_inventory() {
    let index =
        PointSpatialIndex::from_entries_with_checkpoint(points(), &mut || Ok::<_, ()>(())).unwrap();
    let mut work = 0;
    let bounds = Aabb3 {
        min_m: [0.4, 0.002, 0.003],
        max_m: [0.4001, 0.003, 0.005],
    };
    index
        .query_bounds_with_checkpoint(bounds, &mut || {
            work += 1;
            Ok::<_, ()>(())
        })
        .unwrap();
    assert!(work < index.len() / 8, "query work={work}");
}
// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: In-bounds append cancellation exercises every checkpoint; out-of-bounds rebuilding
//   samples the complete checkpoint range including final commit. Failed candidates preserve
//   original point inventory and query answers and resume to the completed answer. No
//   complete-study or measured-memory claim.
#[test]
fn append_and_growth_cancel_at_every_checkpoint_without_partial_inventory() {
    for position in [[0.4, 0.0025, 0.004], [0.44, 0.005, 0.004], [3.0; 3]] {
        let original =
            PointSpatialIndex::from_entries_with_checkpoint(points(), &mut || Ok::<_, ()>(()))
                .unwrap();
        let entry = SpatialPoint::new(position, 12000).unwrap();
        let mut checkpoints = 0;
        let mut completed = original.clone();
        completed
            .insert_with_checkpoint(entry.clone(), &mut || {
                checkpoints += 1;
                Ok::<_, ()>(())
            })
            .unwrap();
        // Out-of-bounds rebuild has many checkpoints: sample its complete range,
        // including final commit. In-bounds append exercises every checkpoint.
        let stride = if position[0] == 3.0 {
            (checkpoints / 100).max(1)
        } else {
            1
        };
        for stop in (0..checkpoints).step_by(stride).chain([checkpoints - 1]) {
            let mut candidate = original.clone();
            let mut visited = 0;
            assert!(candidate
                .insert_with_checkpoint(entry.clone(), &mut || {
                    let fail = visited == stop;
                    visited += 1;
                    if fail {
                        Err(())
                    } else {
                        Ok(())
                    }
                })
                .is_err());
            assert_eq!(candidate.len(), original.len());
            let bounds = Aabb3 {
                min_m: position,
                max_m: position,
            };
            assert_eq!(query(&candidate, bounds), query(&original, bounds));
            candidate
                .insert_with_checkpoint(entry.clone(), &mut || Ok::<_, ()>(()))
                .unwrap();
            assert_eq!(query(&candidate, bounds), query(&completed, bounds));
        }
    }
}
#[test]
fn finite_extreme_and_coincident_points_terminate_and_remain_queryable() {
    assert!(SpatialPoint::new([f64::NAN; 3], 0).is_none());
    let mut entries = (0..100)
        .map(|index| SpatialPoint::new([0.0; 3], index).unwrap())
        .collect::<Vec<_>>();
    entries.push(SpatialPoint::new([f64::MAX; 3], 100).unwrap());
    entries.push(SpatialPoint::new([-f64::MAX; 3], 101).unwrap());
    let index =
        PointSpatialIndex::from_entries_with_checkpoint(entries, &mut || Ok::<_, ()>(())).unwrap();
    assert_eq!(
        query(
            &index,
            Aabb3 {
                min_m: [0.0; 3],
                max_m: [0.0; 3]
            }
        ),
        (0..100).collect::<Vec<_>>()
    );
    assert!(index.clone_with_checkpoint(&mut || Err(())).is_err());
    let copied = index
        .clone_with_checkpoint(&mut || Ok::<_, ()>(()))
        .unwrap();
    assert!(Arc::ptr_eq(
        index.entries.root.as_ref().unwrap(),
        copied.entries.root.as_ref().unwrap()
    ));
    assert!(Arc::ptr_eq(
        index.root.as_ref().unwrap(),
        copied.root.as_ref().unwrap()
    ));
}

#[test]
fn transformed_cloud_queries_and_cancelled_construction_preserve_exhaustive_answers() {
    for scale in [0.001, 1.0, 1000.0] {
        let entries = points()
            .into_iter()
            .map(|entry| {
                let [x, y, z] = entry.position();
                SpatialPoint::new(
                    [
                        scale * (y - x) + 7.0,
                        scale * (x + y) - 2.0,
                        scale * z + 0.5,
                    ],
                    entry.payload,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(
            PointSpatialIndex::from_entries_with_checkpoint(entries.clone(), &mut || Err(()))
                .is_err()
        );
        let index = PointSpatialIndex::from_entries_with_checkpoint(entries.clone(), &mut || {
            Ok::<_, ()>(())
        })
        .unwrap();
        for center in entries.iter().step_by(137) {
            let position = center.position();
            let bounds = Aabb3 {
                min_m: position.map(|value| value - scale * 0.001),
                max_m: position.map(|value| value + scale * 0.001),
            };
            let expected = entries
                .iter()
                .filter(|entry| bounds.contains_point(entry.position()))
                .map(|entry| entry.payload)
                .collect::<Vec<_>>();
            assert_eq!(query(&index, bounds), expected);
        }
    }
}

#[test]
fn distant_outliers_do_not_force_a_dense_cluster_into_a_coarse_leaf() {
    let mut entries = (0..4096)
        .map(|index| SpatialPoint::new([index as f64 * 1e-200, 0.0, 0.0], index).unwrap())
        .collect::<Vec<_>>();
    entries.push(SpatialPoint::new([f64::MAX; 3], 4096).unwrap());
    let index =
        PointSpatialIndex::from_entries_with_checkpoint(entries, &mut || Ok::<_, ()>(())).unwrap();
    let mut work = 0;
    let hits = index
        .query_bounds_with_checkpoint(
            Aabb3 {
                min_m: [0.0; 3],
                max_m: [0.0; 3],
            },
            &mut || {
                work += 1;
                Ok::<_, ()>(())
            },
        )
        .unwrap();
    assert_eq!(
        hits.iter().map(|entry| entry.payload).collect::<Vec<_>>(),
        vec![0]
    );
    assert!(work < 128, "cluster query work={work}");
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Disjoint spatial regions preserve exhaustive original point answers, prune occupied
//   branches and honor cancellation at every visited checkpoint. Traversal component only; no
//   original heat-sink acceptance or whole-task memory claim.
#[test]
fn nonrectangular_region_prunes_branches_preserves_points_and_cancellation() {
    let entries = (0..8000)
        .map(|index| {
            let point = [index % 20, index / 20 % 20, index / 400].map(|value| value as f64 - 9.5);
            SpatialPoint::new(point, index).unwrap()
        })
        .collect::<Vec<_>>();
    let index =
        PointSpatialIndex::from_entries_with_checkpoint(entries.clone(), &mut || Ok::<_, ()>(()))
            .unwrap();
    let bounds = Aabb3 {
        min_m: [-10.0; 3],
        max_m: [10.0; 3],
    };
    let parts = [
        Aabb3 {
            min_m: [-9.5; 3],
            max_m: [-8.5; 3],
        },
        Aabb3 {
            min_m: [8.5; 3],
            max_m: [9.5; 3],
        },
    ];
    let intersects = |box_: Aabb3| parts.iter().any(|part| part.intersects(box_));
    let mut actual = Vec::new();
    let mut work = 0;
    index
        .visit_region_with_checkpoint(
            bounds,
            &intersects,
            &mut |entry| actual.push(entry.payload),
            &mut || {
                work += 1;
                Ok::<_, ()>(())
            },
        )
        .unwrap();
    actual.sort_unstable();
    let expected = entries
        .iter()
        .filter(|entry| {
            parts
                .iter()
                .any(|part| part.contains_point(entry.position()))
        })
        .map(|entry| entry.payload)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
    assert!(work < entries.len() / 10, "region traversal work={work}");
    for stop in 0..work {
        let mut visited = 0;
        assert!(index
            .visit_region_with_checkpoint(bounds, &intersects, &mut |_| {}, &mut || {
                visited += 1;
                if visited > stop {
                    Err(())
                } else {
                    Ok(())
                }
            })
            .is_err());
    }
    assert_eq!(query(&index, bounds).len(), entries.len());
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Conservative point enclosures remain queryable across spatial partitions and after
//   append, rebuild or rejected append. This synthetic index regression preserves broad-phase
//   candidates; it does not independently establish exact physical predicate admission or native
//   study acceptance.
#[test]
fn constructed_enclosures_cross_partition_regions_and_survive_append_rejection() {
    let mut entries = (0..64)
        .map(|id| SpatialPoint::new([id as f64, 0., 0.], id).unwrap())
        .collect::<Vec<_>>();
    entries[0] = SpatialPoint::with_bounds(
        [0.; 3],
        Aabb3 {
            min_m: [0.; 3],
            max_m: [1024., 0., 0.],
        },
        0,
    )
    .unwrap();
    let original =
        PointSpatialIndex::from_entries_with_checkpoint(entries, &mut || Ok::<_, ()>(())).unwrap();
    let region = Aabb3 {
        min_m: [1000., 0., 0.],
        max_m: [1000., 0., 0.],
    };
    assert_eq!(query(&original, region), vec![0]);
    for position in [[512., 0., 0.], [2048., 0., 0.]] {
        let entry = SpatialPoint::new(position, 64).unwrap();
        let mut completed = original.clone();
        completed
            .insert_with_checkpoint(entry.clone(), &mut || Ok::<_, ()>(()))
            .unwrap();
        let expected = completed
            .iter()
            .filter(|entry| entry.bounds().intersects(region))
            .map(|entry| entry.payload)
            .collect::<Vec<_>>();
        assert_eq!(query(&completed, region), expected);
        for limit in 1..40 {
            let mut trial = original.clone();
            let mut work = 0;
            let result = trial.insert_with_checkpoint(entry.clone(), &mut || {
                work += 1;
                if work == limit {
                    Err(())
                } else {
                    Ok(())
                }
            });
            if result.is_err() {
                assert_eq!(trial.len(), 64);
                assert_eq!(query(&trial, region), vec![0]);
            } else {
                assert_eq!(query(&trial, region), expected);
            }
        }
    }
    assert!(SpatialPoint::with_bounds(
        [2.; 3],
        Aabb3 {
            min_m: [0.; 3],
            max_m: [1.; 3]
        },
        0
    )
    .is_none());
}
