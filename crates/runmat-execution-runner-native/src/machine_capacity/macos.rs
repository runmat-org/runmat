//! Darwin counter adapter; never treat installed RAM as available admission capacity.
use std::{io, sync::OnceLock};

// mach2 does not expose this host binding. Keep the Darwin SDK ABI confined to
// the platform adapter instead of depending on libc's deprecated Mach binding.
unsafe extern "C" {
    fn mach_host_self() -> libc::mach_port_t;
}

pub(super) fn available_memory() -> io::Result<u64> {
    static HOST: OnceLock<libc::mach_port_t> = OnceLock::new();
    // Retain one host send right for the process lifetime, not one per refresh.
    // SAFETY: mach_host_self has no arguments or caller-owned output buffers.
    let host = *HOST.get_or_init(|| unsafe { mach_host_self() });
    // SAFETY: sysconf reads a system constant and has no caller-owned buffers.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if host == 0 || page_size <= 0 {
        return Err(io::Error::other("Darwin memory counter discovery failed"));
    }
    // SAFETY: vm_statistics64 contains integer counters; zero is valid for every field.
    let mut counters = unsafe { std::mem::zeroed::<libc::vm_statistics64>() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: the flavor, typed buffer and integer-word count match the Darwin ABI.
    // The kernel writes at most the supplied buffer capacity.
    let status = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&mut counters as *mut libc::vm_statistics64).cast(),
            &mut count,
        )
    };
    if status != libc::KERN_SUCCESS || count < 4 {
        return Err(io::Error::other(format!(
            "Darwin memory counter query failed: status={status}, count={count}"
        )));
    }
    available_bytes(&counters, page_size as u64)
}

/// This process's physical footprint now and its lifetime maximum: resident and compressed dirty
/// memory, the figure the kernel charges it for memory pressure.
pub(super) fn process_footprint() -> io::Result<(u64, u64)> {
    // SAFETY: rusage_info_v4 contains integer counters; zero is valid for every field.
    let mut usage = unsafe { std::mem::zeroed::<libc::rusage_info_v4>() };
    // SAFETY: the flavor matches the typed buffer, which the kernel fills in place.
    let status = unsafe {
        libc::proc_pid_rusage(
            std::process::id() as libc::c_int,
            libc::RUSAGE_INFO_V4,
            (&mut usage as *mut libc::rusage_info_v4).cast(),
        )
    };
    if status != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((usage.ri_phys_footprint, usage.ri_lifetime_max_phys_footprint))
}

fn available_bytes(counters: &libc::vm_statistics64, page_size: u64) -> io::Result<u64> {
    // Free pages already include speculative pages (Darwin vm_statistics.h).
    // Inactive pages are reclaimable estimates, not a guaranteed allocation.
    // Compressor pages occupy a different inventory: subtracting them again can
    // report zero despite free/reclaimable memory. Do not add purgeable overlaps.
    u64::from(counters.free_count)
        .checked_add(u64::from(counters.inactive_count))
        .and_then(|pages| pages.checked_mul(page_size))
        .filter(|_| page_size > 0)
        .ok_or_else(|| io::Error::other("Darwin available-memory counters overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_and_overlapping_counters_do_not_erase_available_pages() {
        let counters = libc::vm_statistics64 {
            free_count: 100,
            inactive_count: 200,
            compressor_page_count: 900,
            speculative_count: 80,
            purgeable_count: 50,
            // SAFETY: all remaining fields are integer counters.
            ..unsafe { std::mem::zeroed() }
        };
        assert_eq!(available_bytes(&counters, 16_384).unwrap(), 4_915_200);
        assert!(available_bytes(&counters, 0).is_err());
        assert!(available_bytes(&counters, u64::MAX).is_err());
    }
}
