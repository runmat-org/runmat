use super::*;

// witness: FEA-SHARED-RESOURCES-001/shared_immutable_inventories @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Point-index trials share original immutable pages. Appending inside prepared bounds
//   at 1024 and 65536 points preserves the original complete inventory and has bounded checkpoint
//   growth without copying every point. This component size series does not qualify arbitrary
//   spatial distributions, full topology trial ownership or native whole-task memory.
#[test]
fn shared_point_append_copies_bounded_pages_and_preserves_original_inventory() {
    let mut costs = Vec::new();
    for count in [1024, 65536] {
        let index = PointSpatialIndex::from_entries_with_checkpoint(
            (0..count)
                .map(|value| SpatialPoint::new([value as f64, 0., 0.], value).unwrap())
                .collect(),
            &mut || Ok::<_, ()>(()),
        )
        .unwrap();
        let mut cost = 0;
        let mut trial = index
            .clone_with_checkpoint(&mut || {
                cost += 1;
                Ok::<_, ()>(())
            })
            .unwrap();
        assert!(Arc::ptr_eq(
            index.entries.root.as_ref().unwrap(),
            trial.entries.root.as_ref().unwrap()
        ));
        assert_eq!(cost, 1);
        trial
            .insert_with_checkpoint(
                SpatialPoint::new([0.25, 0., 0.], count).unwrap(),
                &mut || {
                    cost += 1;
                    Ok::<_, ()>(())
                },
            )
            .unwrap();
        assert_eq!(index.len(), count);
        assert_eq!(trial.len(), count + 1);
        assert!(index
            .iter()
            .zip(trial.iter())
            .all(|(original, retained)| original.payload == retained.payload
                && original.position() == retained.position()));
        assert_eq!(
            query(
                &trial,
                Aabb3 {
                    min_m: [0.25, 0., 0.],
                    max_m: [0.25, 0., 0.]
                }
            ),
            vec![count]
        );
        assert!(
            cost < count / 2,
            "append copied inventory: count={count}, cost={cost}"
        );
        costs.push(cost);
    }
    assert!(costs[1] <= costs[0] * 4, "page-path growth: {costs:?}");
}

#[test]
fn point_pages_preserve_bulk_and_appended_order_across_radix_boundaries() {
    for count in [0, 1, 31, 32, 33, 1023, 1024, 1025, 32767, 32768, 32769] {
        let entries = Entries::from_points(
            (0..count)
                .map(|value| SpatialPoint::new([value as f64; 3], value).unwrap())
                .collect(),
            &mut || Ok::<_, ()>(()),
        )
        .unwrap();
        let original = entries.clone();
        let appended = entries
            .appended(
                SpatialPoint::new([count as f64; 3], count).unwrap(),
                &mut || Ok::<_, ()>(()),
            )
            .unwrap();
        assert_eq!(original.len(), count);
        assert_eq!(
            appended
                .iter()
                .map(|point| point.payload)
                .collect::<Vec<_>>(),
            (0..=count).collect::<Vec<_>>()
        );
        assert_eq!(
            original
                .iter()
                .map(|point| point.payload)
                .collect::<Vec<_>>(),
            (0..count).collect::<Vec<_>>()
        );
        assert!(appended.get(count + 1).is_none());
    }
}
