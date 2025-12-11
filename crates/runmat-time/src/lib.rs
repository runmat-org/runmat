use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(target_arch = "wasm32")]
use js_sys::Date;

#[cfg(target_arch = "wasm32")]
pub use instant::Instant;

#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;

#[cfg(target_arch = "wasm32")]
fn js_duration_since_epoch() -> Duration {
    let millis = Date::now();
    Duration::from_secs_f64(millis / 1000.0)
}

#[cfg(not(target_arch = "wasm32"))]
fn native_duration_since_epoch() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::ZERO)
}

/// Returns a `SystemTime` representing "now" even on wasm targets where
/// `SystemTime::now()` normally panics. On wasm we synthesize the timestamp
/// from `Date.now()` so existing code can keep using standard APIs.
pub fn system_time_now() -> SystemTime {
    #[cfg(target_arch = "wasm32")]
    {
        UNIX_EPOCH + js_duration_since_epoch()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        SystemTime::now()
    }
}

/// Returns the duration since the Unix epoch for the current instant.
pub fn duration_since_epoch() -> Duration {
    #[cfg(target_arch = "wasm32")]
    {
        js_duration_since_epoch()
    }
    #[cfg(not(target_arch = "wasm32"))]
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

