use super::*;
#[test]
fn ordered_union_preserves_arrival_departure_and_unique_ceiling() {
    let s = ProcedureTimeSchedule::merge(&[0., 0.5, 1.], &[0.2, 0.5], &[0.2, 0.65, 1.], 5).unwrap();
    assert_eq!(s.points_s(), &[0., 0.2, 0.5, 0.65, 1.]);
    assert_eq!(s.arrival(1), SignalSampleApproach::LeftLimit);
    assert_eq!(s.departure(1), SignalSampleApproach::RightLimit);
    assert_eq!(s.departure(3), SignalSampleApproach::At);
    assert!(matches!(
        ProcedureTimeSchedule::merge(&[0., 1.], &[], &[0.2, 0.3], 3),
        Err(TimeScheduleError::Ceiling {
            required: 4,
            maximum: 3
        })
    ));
}
#[test]
fn invalid_or_unsorted_sources_cannot_create_negative_intervals() {
    for source in [
        &[0.7, 0.2][..],
        &[f64::NAN],
        &[f64::INFINITY],
        &[-0.1],
        &[1.1],
    ] {
        assert!(ProcedureTimeSchedule::merge(&[0., 1.], source, &[], 9).is_err());
        assert!(ProcedureTimeSchedule::merge(&[0., 1.], &[], source, 9).is_err());
    }
    for grid in [&[][..], &[0.], &[1., 0.], &[0., 0.]] {
        assert!(ProcedureTimeSchedule::merge(grid, &[], &[], 9).is_err());
    }
    let s = ProcedureTimeSchedule::merge(&[-0., 1.], &[0., 0.], &[-0.], 2).unwrap();
    assert_eq!(s.points_s()[0].to_bits(), 0_f64.to_bits());
}

#[test]
fn counted_union_matches_an_independent_set_for_all_small_source_inventories() {
    use std::collections::BTreeSet;
    let grid = [0., 0.5, 1.];
    for events in 0..32u32 {
        for required in 0..32u32 {
            let source = |mask| {
                (0..5)
                    .filter(|i| mask & (1 << i) != 0)
                    .flat_map(|i| [i as f64 / 4., i as f64 / 4.])
                    .collect::<Vec<_>>()
            };
            let events = source(events);
            let required = source(required);
            let expected = grid
                .iter()
                .chain(&events)
                .chain(&required)
                .map(|v| (*v * 4.) as u32)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|v| v as f64 / 4.)
                .collect::<Vec<_>>();
            let schedule =
                ProcedureTimeSchedule::merge(&grid, &events, &required, expected.len()).unwrap();
            assert_eq!(schedule.points_s(), expected);
            for (i, point) in expected.iter().enumerate() {
                assert_eq!(
                    schedule.arrival(i),
                    if events.contains(point) {
                        SignalSampleApproach::LeftLimit
                    } else {
                        SignalSampleApproach::At
                    }
                );
            }
        }
    }
}

#[test]
fn exact_heap_inventory_and_both_ceilings_reject_before_allocation() {
    let grid = [0., 0.5, 1.];
    let events = [0.2, 0.2, 0.5];
    let required = [0.2, 0.65, 1.];
    let bytes =
        ProcedureTimeSchedule::construction_resident_bytes(&grid, &events, &required, 5).unwrap();
    assert_eq!(bytes, 7 * std::mem::size_of::<f64>() as u64);
    let (schedule, retained, peak) = crate::topology::measure_test_allocations(|| {
        ProcedureTimeSchedule::merge_admitted(&grid, &events, &required, 5, bytes)
    });
    assert_eq!(schedule.unwrap().points_s(), &[0., 0.2, 0.5, 0.65, 1.]);
    assert_eq!(peak as u64, bytes);
    assert_eq!(retained, peak);
    for (samples, resident) in [(5, bytes - 1), (4, bytes)] {
        let (result, retained, peak) = crate::topology::measure_test_allocations(|| {
            ProcedureTimeSchedule::merge_admitted(&grid, &events, &required, samples, resident)
        });
        assert!(result.is_err());
        assert_eq!((retained, peak), (0, 0));
    }
    assert!(super::union::resident_bytes(usize::MAX, 1).is_err());
}
