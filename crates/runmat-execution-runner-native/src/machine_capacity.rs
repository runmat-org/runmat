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

/// Execution ceilings for one local meshing run: sized from the memory this host admits for work
/// (the same authority as publication staging and the thread plan) and from the requested mesh
/// size. They bound whether the run completes, never what it produces.
pub fn meshing_execution_ceilings(
    limits: &runmat_meshing_core::MeshingSizeLimits,
) -> runmat_meshing_core::MeshingExecutionCeilings {
    meshing_execution_ceilings_for(memory_bytes().unwrap_or(0), limits)
}

/// Ceilings for a run admitted `capacity_bytes` of memory: the pool's own admitted capacity,
/// lowered to the run's authored per-worker memory when one is set, so every task the run submits
/// fits what the run is admitted to use. The run carries no wall-time budget for meshing, so the
/// wall-time ceiling is unbounded.
pub fn meshing_execution_ceilings_for(
    capacity_bytes: u64,
    limits: &runmat_meshing_core::MeshingSizeLimits,
) -> runmat_meshing_core::MeshingExecutionCeilings {
    runmat_meshing_core::MeshingExecutionCeilings::for_host(capacity_bytes, limits)
}

/// Default host budget for staging one study publication: a quarter of the memory this host
/// admits for work, at least 1 GiB. An explicitly configured budget takes precedence.
pub fn publication_staging_bytes() -> u64 {
    const MINIMUM: u64 = 1024 * 1024 * 1024;
    memory_bytes().map_or(MINIMUM, |bytes| (bytes / 4).max(MINIMUM))
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

/// The single thread-plan authority for local FEA execution.
///
/// `workers` co-located task slots share the machine: with no explicit thread count each slot
/// gets an equal share of the cores (`capacity / workers`, at least one), so concurrent workers
/// never oversubscribe the host. An explicit `cpu_threads_per_worker` is honored only when all
/// co-located workers fit (`workers × threads ≤ capacity`). Sweep case concurrency and remote
/// fan-out derive from the same plan ([`case_concurrency`]).
pub(super) fn study_allocation(workers: u32, explicit_threads: Option<u32>) -> Result<u32, String> {
    allocate(fea_cpu_millicores() / 1000, workers, explicit_threads)
}

/// Numerical defaults fit both this machine and the original admitted FEA root.
/// This is a planning ceiling, not a mutation of an authored CPU/thread request.
fn fea_cpu_millicores() -> u32 {
    admitted_capacity(
        cpu_millicores(),
        runmat_runtime::analysis::fea_run_admission::current_fea_cpu_capacity_millicores(),
    )
}

fn admitted_capacity(machine: u32, original: Option<u64>) -> u32 {
    original.map_or(machine, |original| {
        machine.min(original.min(u64::from(u32::MAX)) as u32)
    })
}

fn allocate(capacity: u32, workers: u32, explicit_threads: Option<u32>) -> Result<u32, String> {
    if workers == 0 || workers > capacity {
        return Err("requested local worker count exceeds available CPU capacity".into());
    }
    let threads = match explicit_threads {
        Some(threads) => threads,
        None => (capacity / workers).max(1),
    };
    if threads == 0 || threads.saturating_mul(workers) > capacity {
        return Err(
            "requested local worker/thread allocation exceeds available CPU capacity".into(),
        );
    }
    Ok(threads)
}

/// Independent cases (sweep cases) that may run at once on this host given the per-case plan:
/// `capacity / (workers × threads)`, at least one, at most `cases` and the authored limit.
pub(crate) fn case_concurrency(
    workers_per_case: u32,
    threads_per_worker: u32,
    cases: usize,
    authored_limit: Option<usize>,
) -> usize {
    let capacity = (fea_cpu_millicores() / 1000).max(1) as usize;
    let per_case =
        (workers_per_case.max(1) as usize).saturating_mul(threads_per_worker.max(1) as usize);
    (capacity / per_case)
        .max(1)
        .min(authored_limit.unwrap_or(usize::MAX))
        .min(cases.max(1))
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
        // Co-located workers split the cores instead of oversubscribing.
        assert_eq!(allocate(16, 4, None).unwrap(), 4);
        assert_eq!(allocate(16, 5, None).unwrap(), 3);
        assert_eq!(allocate(16, 3, Some(2)).unwrap(), 2);
        assert_eq!(allocate(16, 4, Some(4)).unwrap(), 4);
        assert!(allocate(16, 4, Some(5)).is_err());
        assert!(allocate(16, 4, Some(17)).is_err());
        assert!(allocate(16, 17, None).is_err());
        assert!(allocate(16, 0, None).is_err());
    }

    #[test]
    fn trusted_root_caps_defaults_and_rejects_explicit_excess() {
        let cap = admitted_capacity(32_000, Some(2_000)) / 1000;
        assert_eq!(allocate(cap, 1, None).unwrap(), 2);
        assert_eq!(allocate(cap, 2, None).unwrap(), 1);
        assert!(allocate(cap, 1, Some(3)).is_err());
        assert!(allocate(cap, 3, None).is_err());
        assert!(allocate(admitted_capacity(32_000, Some(999)) / 1000, 1, None).is_err());
        assert_eq!(admitted_capacity(32_000, None), 32_000);
    }
}
