use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use js_sys::Date;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub use instant::Instant;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub use std::time::Instant;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn js_duration_since_epoch() -> Duration {
    let millis = Date::now();
    Duration::from_secs_f64(millis / 1000.0)
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn native_duration_since_epoch() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
}

/// Returns the current wall clock. Bare WebAssembly uses `Date.now()`;
/// Emscripten and other standard-library hosts use their standard clock.
pub fn system_time_now() -> SystemTime {
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        UNIX_EPOCH + js_duration_since_epoch()
    }
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        SystemTime::now()
    }
}

/// Returns the duration since the Unix epoch for the current instant.
pub fn duration_since_epoch() -> Duration {
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        js_duration_since_epoch()
    }
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        native_duration_since_epoch()
    }
}

/// Milliseconds since the Unix epoch, safe to call on wasm and native hosts.
pub fn unix_timestamp_ms() -> u128 {
    duration_since_epoch().as_millis()
}

/// Microseconds since the Unix epoch.
pub fn unix_timestamp_us() -> u128 {
    duration_since_epoch().as_micros()
}

/// Nanoseconds since the Unix epoch.
pub fn unix_timestamp_ns() -> u128 {
    duration_since_epoch().as_nanos()
}

/// Converts a duration to nanoseconds without truncating on overflow.
pub fn duration_ns_saturating(duration: Duration) -> u64 {
    duration.as_nanos().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duration_nanoseconds_saturate() {
        assert_eq!(duration_ns_saturating(Duration::from_nanos(42)), 42);
        assert_eq!(duration_ns_saturating(Duration::MAX), u64::MAX);
    }
}
