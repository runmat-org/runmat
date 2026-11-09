//! OS peak-resident diagnostics for snapshot phase logs, separate from allocation accounting.

struct PeakResidentMemory(Option<u64>);

impl std::fmt::Display for PeakResidentMemory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            Some(mib) => write!(formatter, "{mib} MiB"),
            None => formatter.write_str("unavailable"),
        }
    }
}

pub(in super::super) fn peak_resident_memory() -> impl std::fmt::Display {
    PeakResidentMemory(peak_resident_bytes().map(|bytes| bytes / (1024 * 1024)))
}

#[cfg(unix)]
fn peak_resident_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    // SAFETY: the output points to writable storage for one rusage value.
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: successful getrusage initialized the output value.
    let maximum = u64::try_from(unsafe { usage.assume_init() }.ru_maxrss).ok()?;
    if cfg!(target_os = "macos") {
        Some(maximum)
    } else {
        maximum.checked_mul(1024)
    }
}

#[cfg(windows)]
fn peak_resident_bytes() -> Option<u64> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size,
        ..Default::default()
    };
    // SAFETY: the pseudo-handle names this process, counters is writable, and
    // size matches its fixed C layout. The pseudo-handle does not need closing.
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) } == 0 {
        return None;
    }
    u64::try_from(counters.PeakWorkingSetSize).ok()
}

#[cfg(not(any(unix, windows)))]
fn peak_resident_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peak_diagnostics_preserve_availability_and_units() {
        assert_eq!(PeakResidentMemory(None).to_string(), "unavailable");
        assert_eq!(PeakResidentMemory(Some(42)).to_string(), "42 MiB");
        #[cfg(any(unix, windows))]
        assert!(peak_resident_bytes().is_some_and(|bytes| bytes > 0));
    }
}
