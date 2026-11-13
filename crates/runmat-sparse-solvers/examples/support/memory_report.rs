//! Optional process-lifetime RSS for benchmark JSON, including process startup.

use std::fmt;

pub struct PeakRssGiB(Option<u64>);

impl fmt::Display for PeakRssGiB {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            Some(bytes) => fmt::Display::fmt(&(bytes as f64 / (1u64 << 30) as f64), formatter),
            None => formatter.write_str("null"),
        }
    }
}

pub fn peak_rss_gib() -> PeakRssGiB {
    PeakRssGiB(process_peak_rss_bytes())
}

#[cfg(unix)]
fn process_peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` is writable storage for one `rusage` value.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: successful `getrusage` initialized the complete value.
    let maximum = u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).ok()?;
    if cfg!(target_os = "macos") {
        Some(maximum)
    } else {
        maximum.checked_mul(1024)
    }
}

#[cfg(not(unix))]
fn process_peak_rss_bytes() -> Option<u64> {
    None
}
