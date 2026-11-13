use crate::{Aabb3, SpatialEntry, UniformGridSpatialIndex};

fn point(value: u32, coordinates: [f64; 3]) -> SpatialEntry<u32> {
    SpatialEntry {
        payload: value,
        bounds: Aabb3 {
            min_m: coordinates,
            max_m: coordinates,
        },
    }
}

#[test]
fn append_and_bounds_queries_match_exhaustive_inventory_including_growth() {
    let mut entries = vec![point(0, [-1.0; 3]), point(1, [1.0; 3])];
    let mut index = UniformGridSpatialIndex::from_entries(entries.clone());
    for value in 2..100 {
        let coordinates = [
            f64::from(value % 13) / 7.0 - 0.5,
            f64::from(value % 7) - 3.0,
            f64::from(value % 5),
        ];
        let entry = point(value, coordinates);
        entries.push(entry.clone());
        index
            .insert_with_checkpoint(entry, &mut || Ok::<_, ()>(()))
            .unwrap();
        for query in [
            Aabb3 {
                min_m: [-0.25; 3],
                max_m: [0.25; 3],
            },
            Aabb3 {
                min_m: coordinates,
                max_m: coordinates,
            },
            Aabb3 {
                min_m: [-4.0; 3],
                max_m: [4.0; 3],
            },
        ] {
            let actual = index
                .query_bounds_with_checkpoint(query, &mut || Ok::<_, ()>(()))
                .unwrap()
                .into_iter()
                .map(|entry| entry.payload)
                .collect::<Vec<_>>();
            let expected = entries
                .iter()
                .filter(|entry| entry.bounds.intersects(query))
                .map(|entry| entry.payload)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn cancellation_during_append_preserves_complete_original_index() {
    let original =
        UniformGridSpatialIndex::from_entries(vec![point(0, [-1.0; 3]), point(1, [1.0; 3])]);
    for coordinates in [[0.0; 3], [3.0; 3]] {
        let mut succeeded = false;
        for allowance in 0..100 {
            let mut candidate = original.clone();
            let mut remaining = allowance;
            let result = candidate.insert_with_checkpoint(point(2, coordinates), &mut || {
                if remaining == 0 {
                    Err(())
                } else {
                    remaining -= 1;
                    Ok(())
                }
            });
            if result.is_ok() {
                assert_eq!(candidate.entries().len(), 3);
                assert_eq!(candidate.query_point(coordinates)[0].payload, 2);
                succeeded = true;
                break;
            }
            assert_eq!(candidate, original);
        }
        assert!(succeeded);
    }
    assert!(UniformGridSpatialIndex::from_entries_with_checkpoint(
        original.entries().to_vec(),
        &mut || Err(())
    )
    .is_err());
    assert!(original
        .query_bounds_with_checkpoint(original.bounds(), &mut || Err(()))
        .is_err());
}

#[test]
fn dense_point_query_accounts_local_candidates_instead_of_whole_inventory() {
    let mut entries = Vec::new();
    for x in 0..40 {
        for y in 0..40 {
            for z in 0..40 {
                entries.push(point(
                    entries.len() as u32,
                    [f64::from(x), f64::from(y), f64::from(z)],
                ));
            }
        }
    }
    let index = UniformGridSpatialIndex::from_entries(entries);
    let mut steps = 0;
    let result = index
        .query_bounds_with_checkpoint(
            Aabb3 {
                min_m: [19.9; 3],
                max_m: [20.1; 3],
            },
            &mut || {
                steps += 1;
                Ok::<_, ()>(())
            },
        )
        .unwrap();
    assert_eq!(result.len(), 1);
    assert!(steps < 100, "local grid query consumed {steps} steps");
}

#[test]
fn cooperative_copy_preserves_inventory_and_cancels_at_each_retained_step() {
    let original = UniformGridSpatialIndex::from_entries(
        (0..64)
            .map(|index| point(index, [f64::from(index); 3]))
            .collect(),
    );
    let mut completed = false;
    for allowance in 0..300 {
        let mut remaining = allowance;
        let copied = original.clone_with_checkpoint(&mut || {
            if remaining == 0 {
                Err(())
            } else {
                remaining -= 1;
                Ok(())
            }
        });
        if let Ok(copied) = copied {
            assert_eq!(copied, original);
            assert_eq!(
                copied.query_bounds(original.bounds()),
                original.query_bounds(original.bounds())
            );
            completed = true;
            break;
        }
        assert_eq!(original.entries().len(), 64);
    }
    assert!(completed);
}
