use super::*;

#[test]
fn union_counts_events_and_observations_once_and_keeps_left_limits() {
    let schedule =
        ThermalSolveSchedule::merge(&[0.0, 0.5, 1.0], &[0.2, 0.5], &[0.2, 0.65, 1.0], 5).unwrap();
    assert_eq!(schedule.points_s(), &[0.0, 0.2, 0.5, 0.65, 1.0]);
    assert_eq!(schedule.sample_count(), 5);
    assert_eq!(schedule.solve_count(), 4);
    assert_eq!(schedule.approach(1), SignalSampleApproach::LeftLimit);
    assert_eq!(schedule.approach(2), SignalSampleApproach::LeftLimit);
    assert_eq!(schedule.approach(3), SignalSampleApproach::At);
    assert!(ThermalSolveSchedule::merge(&[0.0, 0.5, 1.0], &[0.2], &[0.65], 4).is_err());
}

#[test]
fn signed_zero_does_not_create_a_zero_duration_increment() {
    let schedule = ThermalSolveSchedule::merge(&[-0.0, 1.0], &[0.0], &[-0.0], 2).unwrap();
    assert_eq!(schedule.points_s(), &[0.0, 1.0]);
    assert_eq!(schedule.solve_count(), 1);
    assert_eq!(schedule.approach(0), SignalSampleApproach::LeftLimit);
}

#[test]
fn out_of_horizon_and_nonfinite_samples_fail_closed() {
    for invalid in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(ThermalSolveSchedule::merge(&[0.0, 1.0], &[], &[invalid], 3).is_err());
    }
}

mod resources;
