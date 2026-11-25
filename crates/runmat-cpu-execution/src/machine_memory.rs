//! The physical memory of the machine or container this process runs on, and what this process
//! holds of it. Hosts and the runtime read the same measurement, so every memory limit a local run
//! applies derives from one probe.
use std::io;

#[cfg(target_os = "macos")]
mod macos;

/// The memory this machine admits for work now: its available physical memory (or the
/// container's, whichever is smaller), less a quarter left for the OS and other work.
/// Admission is a snapshot; runtime ceilings and cancellation remain authoritative. A machine
/// whose available memory cannot be read gets an error; no portable constant stands in for it.
pub fn admissible_memory_bytes() -> io::Result<u64> {
    let available = available_memory_bytes()?;
    let budget = available - available / 4;
    if budget == 0 {
        Err(io::Error::other(
            "local memory discovery returned no available capacity",
        ))
    } else {
        Ok(budget)
    }
}

/// This process's physical footprint.
pub fn process_footprint_bytes() -> io::Result<u64> {
    process_footprint_and_peak_bytes().map(|(bytes, _)| bytes)
}

/// This process's physical footprint now and its lifetime maximum. On macOS this is resident
/// and compressed dirty memory, the figure the kernel charges the process for memory pressure;
/// elsewhere it is resident memory, reported for both.
#[cfg(target_os = "macos")]
pub fn process_footprint_and_peak_bytes() -> io::Result<(u64, u64)> {
    macos::process_footprint()
}

#[cfg(not(target_os = "macos"))]
pub fn process_footprint_and_peak_bytes() -> io::Result<(u64, u64)> {
    let pid = sysinfo::get_current_pid().map_err(io::Error::other)?;
    let mut system = sysinfo::System::new();
    system.refresh_process_specifics(pid, sysinfo::ProcessRefreshKind::new().with_memory());
    system
        .process(pid)
        .map(|process| (process.memory(), process.memory()))
        .ok_or_else(|| io::Error::other("this process's resident memory is not readable"))
}

/// Physical memory available to new work on this machine, bounded by the container's free
/// memory when the process runs in one.
#[cfg(target_os = "macos")]
pub fn available_memory_bytes() -> io::Result<u64> {
    macos::available_memory()
}

#[cfg(not(target_os = "macos"))]
pub fn available_memory_bytes() -> io::Result<u64> {
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
    fn this_machine_admits_memory_and_this_process_holds_some() {
        let admissible = admissible_memory_bytes().unwrap();
        assert!(admissible > 0 && admissible <= available_memory_bytes().unwrap());
        assert!(process_footprint_bytes().unwrap() > 0);
    }
}
