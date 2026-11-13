//! Process-image RSS diagnostics, not allocation or scheduler-reservation accounting.

#[cfg(target_os = "linux")]
pub fn process_peak_rss_bytes() -> Option<u64> {
    // Linux rusage survives exec and can retain the previous image's peak.
    // VmHWM belongs to the current memory image, which an isolated solve measures.
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let value = status
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?;
    let mut fields = value.split_whitespace();
    let kibibytes = fields.next()?.parse::<u64>().ok()?;
    if fields.next()? != "kB" || fields.next().is_some() {
        return None;
    }
    kibibytes.checked_mul(1024)
}

#[cfg(all(unix, not(target_os = "linux")))]
pub fn process_peak_rss_bytes() -> Option<u64> {
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
pub fn process_peak_rss_bytes() -> Option<u64> {
    None
}
