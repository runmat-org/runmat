//! Diagnostic-binary-local Rust heap peak measurement; native allocations are not included.

#[path = "peak_allocator/counting.rs"]
mod counting;

pub use counting::{begin_measurement, end_measurement};

#[cfg(unix)]
pub fn process_peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` is writable storage for one `rusage` value.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: successful `getrusage` initialized the complete value.
    let maximum = u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).ok()?;
    Some(if cfg!(target_os = "macos") {
        maximum
    } else {
        maximum.saturating_mul(1024)
    })
}

#[cfg(not(unix))]
pub fn process_peak_rss_bytes() -> Option<u64> {
    None
}
