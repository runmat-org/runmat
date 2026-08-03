use std::time::Duration;

use runmat_execution_transport_native::control::{HeartbeatSchedule, ReconnectBackoff};

#[test]
fn reconnect_backoff_is_bounded_and_resettable() {
    let mut backoff =
        ReconnectBackoff::new(Duration::from_secs(1), Duration::from_secs(4)).unwrap();
    assert_eq!(backoff.next_delay(), Duration::from_secs(1));
    assert_eq!(backoff.next_delay(), Duration::from_secs(2));
    assert_eq!(backoff.next_delay(), Duration::from_secs(4));
    assert_eq!(backoff.next_delay(), Duration::from_secs(4));
    backoff.reset();
    assert_eq!(backoff.next_delay(), Duration::from_secs(1));
}

#[test]
fn heartbeat_requires_time_for_reconnect_and_clock_skew() {
    assert!(HeartbeatSchedule::new(
        Duration::from_secs(10),
        Duration::from_secs(30),
        Duration::from_secs(5)
    )
    .is_ok());
    assert!(HeartbeatSchedule::new(
        Duration::from_secs(20),
        Duration::from_secs(20),
        Duration::from_secs(1)
    )
    .is_err());
}
