use super::test_control::Budget;
use super::*;
use crate::model::{
    exact_evaluator::tests::registry,
    exact_topology_tests::{model, topology},
};
#[test]
fn exact_validation_releases_all_scratch_between_queries_and_rejects_short_admission() {
    let topology = topology();
    let model = model();
    let registry = registry();
    for evaluators in [false, true] {
        let validate = |control: &Budget| {
            if evaluators {
                registry.validate_against_with_control(&topology, &model, control)
            } else {
                topology.validate_against_with_control(&model, control)
            }
        };
        let control = Budget::new(1 << 20);
        validate(&control).unwrap();
        let peak = control.peak();
        assert!(peak > 0);
        assert_eq!(control.live(), 0);
        for _ in 0..64 {
            validate(&control).unwrap();
            assert_eq!(control.live(), 0);
            assert_eq!(control.peak(), peak);
        }
        validate(&Budget::new(peak)).unwrap();
        let denied = Budget::new(peak - 1);
        assert!(validate(&denied).is_err());
        assert_eq!(denied.live(), 0);
        let interrupted = Budget::with_stop(1 << 20, 2);
        assert!(validate(&interrupted).is_err());
        assert_eq!(interrupted.live(), 0);
    }
}
#[test]
fn typed_validation_buffers_deny_before_backing_allocation_and_release_on_unwind() {
    let control = Budget::new(64);
    assert!(ValidationWork(Some(&control)).buffer::<u64>(9).is_err());
    assert_eq!(control.live(), 0);
    {
        let mut values = ValidationWork(Some(&control)).buffer::<u64>(8).unwrap();
        for value in 0..8 {
            values.push(value);
        }
        assert_eq!(control.live(), 64);
        assert_eq!(&*values, &[0, 1, 2, 3, 4, 5, 6, 7]);
    }
    assert_eq!(control.live(), 0);
}
