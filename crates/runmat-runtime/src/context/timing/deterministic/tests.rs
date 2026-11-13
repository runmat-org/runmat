use super::*;
use runmat_execution::{
    cancellation::CancellationSource,
    host::{NativeClockCpuInputs, NATIVE_CLOCK_INPUTS_SCHEMA_VERSION},
    CancellationReason,
};
use std::task::{Context, Poll};

fn inputs() -> NativeClockInputs {
    NativeClockInputs {
        schema_version: NATIVE_CLOCK_INPUTS_SCHEMA_VERSION,
        initial_monotonic_nanos: 5,
        initial_wall_unix_seconds: 1_700_000_000,
        initial_wall_nanosecond: 123,
        read_step_nanos: 10_000_000,
        local_offset_seconds: 19_800,
        process_cpu: Some(NativeClockCpuInputs {
            initial_nanos: 2_000_000_000,
            read_step_nanos: 25_000_000,
        }),
    }
}

#[test]
fn replay_advances_monotonic_wall_calendar_and_explicit_cpu_without_host_sampling() {
    let source = CancellationSource::new();
    let first = DeterministicTimingService::new(inputs(), source.token()).unwrap();
    let second = DeterministicTimingService::new(inputs(), source.token()).unwrap();
    // Profiling setup may probe repeatedly, but cannot consume language time.
    for _ in 0..4 {
        assert!(first.command_thread_cpu_sampler().is_none());
    }
    for clock in [&first, &second] {
        assert_eq!(clock.monotonic_elapsed(), Duration::from_nanos(5));
        let wall = clock.wall_time_since_epoch();
        assert_eq!(wall, Duration::new(1_700_000_000, 10_000_123));
        let instant = RuntimeCalendarInstant::from_epoch_duration(wall).unwrap();
        let local = clock.local_datetime_at_instant(instant).unwrap();
        assert_eq!(clock.local_datetime_to_instant(local), Some(instant));
        let utc = chrono::DateTime::from_timestamp(1_700_000_000, 10_000_123)
            .unwrap()
            .naive_utc();
        assert_eq!(local.signed_duration_since(utc).num_seconds(), 19_800);
        assert!(clock
            .local_datetime_to_instant(chrono::DateTime::from_timestamp(0, 0).unwrap().naive_utc())
            .is_some());
        assert_eq!(clock.process_cpu_seconds(), Some(2.0));
        assert_eq!(clock.process_cpu_seconds(), Some(2.025));
        assert!(
            clock.command_thread_cpu_sampler().is_none(),
            "logical CPU is not physical profiling evidence"
        );
        futures::executor::block_on(clock.sleep(Duration::from_secs(2))).unwrap();
        assert_eq!(
            clock.monotonic_elapsed(),
            Duration::from_nanos(2_020_000_005)
        );
    }
    assert_eq!(first.state.get().elapsed, second.state.get().elapsed);
    let mut absent = inputs();
    absent.process_cpu = None;
    assert!(DeterministicTimingService::new(absent, source.token())
        .unwrap()
        .process_cpu_seconds()
        .is_none());
}

#[test]
fn cold_pending_cancelled_and_overflowing_waits_never_advance_abandoned_time() {
    let source = CancellationSource::new();
    let clock = DeterministicTimingService::new(inputs(), source.token()).unwrap();
    let before = clock.state.get().elapsed;
    drop(clock.sleep(Duration::from_secs(7)));
    assert_eq!(clock.state.get().elapsed, before);
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    let mut pending = clock.sleep(Duration::from_secs(7));
    assert!(pending.as_mut().poll(&mut context).is_pending());
    drop(pending);
    assert_eq!(clock.state.get().elapsed, before);
    let mut pending = clock.sleep(Duration::from_secs(7));
    assert!(pending.as_mut().poll(&mut context).is_pending());
    source.cancel(CancellationReason::User);
    assert!(matches!(
        pending.as_mut().poll(&mut context),
        Poll::Ready(Err(_))
    ));
    assert_eq!(clock.state.get().elapsed, before);
    assert!(futures::executor::block_on(clock.sleep(Duration::ZERO)).is_err());
    let fresh =
        DeterministicTimingService::new(inputs(), CancellationSource::new().token()).unwrap();
    assert!(futures::executor::block_on(fresh.sleep(Duration::MAX)).is_err());
    assert_eq!(fresh.state.get().elapsed, Duration::ZERO);
    futures::executor::block_on(fresh.sleep(Duration::from_secs(3))).unwrap();
    assert_eq!(fresh.state.get().elapsed, Duration::from_secs(3));
    let weak = Rc::downgrade(&fresh.state);
    let pending = fresh.sleep(Duration::from_secs(1));
    drop(fresh);
    assert!(
        weak.upgrade().is_some(),
        "the cold future owns its exact state"
    );
    drop(pending);
    assert!(
        weak.upgrade().is_none(),
        "abandonment retires its final state owner"
    );
}

#[test]
fn invalid_clock_policy_rejects_before_creating_a_service() {
    for invalid in [0, 86_400_000_000_001] {
        let mut policy = inputs();
        policy.read_step_nanos = invalid;
        assert!(
            DeterministicTimingService::new(policy, CancellationSource::new().token()).is_err()
        );
    }
    let mut invalid = inputs();
    invalid.initial_wall_nanosecond = 1_000_000_000;
    assert!(invalid.validate().is_err());
    invalid = inputs();
    invalid.local_offset_seconds = 86_400;
    assert!(invalid.validate().is_err());
}

#[test]
fn fixed_calendar_preserves_signed_and_leap_instants_without_consuming_time() {
    let clock =
        DeterministicTimingService::new(inputs(), CancellationSource::new().token()).unwrap();
    for instant in [
        RuntimeCalendarInstant::from_unix_timestamp(-60, 123).unwrap(),
        RuntimeCalendarInstant::from_unix_timestamp(1_483_228_799, 1_500_000_000).unwrap(),
    ] {
        let local = clock.local_datetime_at_instant(instant).unwrap();
        assert_eq!(clock.local_datetime_to_instant(local), Some(instant));
    }
    assert_eq!(clock.state.get().elapsed, Duration::ZERO);
}
