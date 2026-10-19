//! Local host discovery, separated from portable numerical policies.
use std::io;

#[cfg(target_os = "macos")]
mod macos;

pub(crate) fn cpu_millicores() -> u32 {
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(runmat_cpu_execution::MAX_THREADS);
    threads as u32 * 1000
}

pub(crate) fn memory_bytes() -> io::Result<u64> {
    // Leave a quarter of currently available memory for the OS and other work.
    // Admission is a snapshot; runtime ceilings/cancellation remain authoritative.
    let available = available_memory()?;
    let budget = available - available / 4;
    if budget == 0 {
        Err(io::Error::other(
            "local memory discovery returned no available capacity",
        ))
    } else {
        Ok(budget)
    }
}

#[cfg(target_os = "macos")]
fn available_memory() -> io::Result<u64> {
    macos::available_memory()
}

#[cfg(not(target_os = "macos"))]
fn available_memory() -> io::Result<u64> {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    Ok(usable_memory(
        system.available_memory(),
        system.cgroup_limits().map(|limit| limit.free_memory),
    ))
}

#[cfg(any(not(target_os = "macos"), test))]
fn usable_memory(host_available: u64, container_available: Option<u64>) -> u64 {
    container_available.map_or(host_available, |available| host_available.min(available))
}

pub(super) fn study_allocation(workers: u32, explicit_threads: Option<u32>) -> Result<u32, String> {
    allocate(cpu_millicores() / 1000, workers, explicit_threads)
}
fn allocate(capacity: u32, workers: u32, explicit_threads: Option<u32>) -> Result<u32, String> {
    if workers == 0 || workers > capacity {
        return Err("requested local worker count exceeds available CPU capacity".into());
    }
    // Worker count is a slot ceiling, not a simultaneous CPU reservation.
    // Wide tasks may use the whole budget; aggregate pool admission governs fanout.
    let threads = explicit_threads.unwrap_or(capacity);
    if threads == 0 || threads > capacity {
        return Err(
            "requested local worker/thread allocation exceeds available CPU capacity".into(),
        );
    }
    Ok(threads)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn container_memory_cannot_advertise_host_capacity() {
        assert_eq!(usable_memory(16_000, Some(2_000)), 2_000);
        assert_eq!(usable_memory(2_000, Some(16_000)), 2_000);
        assert_eq!(usable_memory(16_000, None), 16_000);
        assert_eq!(usable_memory(16_000, Some(0)), 0);
    }
    #[test]
    fn automatic_and_explicit_study_allocations_share_machine_capacity() {
        assert_eq!(allocate(16, 1, None).unwrap(), 16);
        assert_eq!(allocate(16, 4, None).unwrap(), 16);
        assert_eq!(allocate(16, 3, Some(2)).unwrap(), 2);
        assert_eq!(allocate(16, 4, Some(5)).unwrap(), 5);
        assert!(allocate(16, 4, Some(17)).is_err());
        assert!(allocate(16, 17, None).is_err());
        assert!(allocate(16, 0, None).is_err());
    }
}
