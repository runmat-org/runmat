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

/// The memory this process admits for work: three quarters of its pool, leaving a quarter for
/// the OS and other work. The pool is the memory available now plus the memory this process
/// already holds. Every requirement law charges the run's own resident components (mesh, model,
/// compiled properties), so they count inside the budget; reading only the available memory
/// would subtract them a second time. The pool does not change as the run allocates, so path
/// selection, the early checks and the worker envelope read the same budget whenever they ask.
/// Admission is a snapshot of other processes' use; runtime ceilings and cancellation remain
/// authoritative.
pub(crate) fn memory_bytes() -> io::Result<u64> {
    admitted_memory(available_memory()?, process_resident_memory()?)
}

fn admitted_memory(available: u64, resident: u64) -> io::Result<u64> {
    let pool = available.saturating_add(resident);
    let budget = pool - pool / 4;
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

#[cfg(target_os = "macos")]
fn process_resident_memory() -> io::Result<u64> {
    macos::process_footprint()
}

#[cfg(not(target_os = "macos"))]
fn process_resident_memory() -> io::Result<u64> {
    let pid = sysinfo::get_current_pid().map_err(io::Error::other)?;
    let mut system = sysinfo::System::new();
    system.refresh_process_specifics(pid, sysinfo::ProcessRefreshKind::new().with_memory());
    system
        .process(pid)
        .map(sysinfo::Process::memory)
        .ok_or_else(|| io::Error::other("this process's resident memory is not readable"))
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
    fn memory_this_process_holds_counts_inside_its_budget() {
        // 20 GB available while the process holds 10 GB: the pool is 30 GB, not 20 GB.
        const GB: u64 = 1_000_000_000;
        assert_eq!(admitted_memory(20 * GB, 10 * GB).unwrap(), 22_500_000_000);
        // The budget is the same before and after the process allocates from the pool.
        assert_eq!(
            admitted_memory(30 * GB, 0).unwrap(),
            admitted_memory(20 * GB, 10 * GB).unwrap()
        );
        assert!(admitted_memory(0, 0).is_err());
        assert_eq!(admitted_memory(u64::MAX, u64::MAX).unwrap(), u64::MAX - u64::MAX / 4);
    }
    #[test]
    fn this_process_footprint_is_readable_and_counts() {
        assert!(process_resident_memory().unwrap() > 0);
        assert!(memory_bytes().unwrap() >= available_memory().unwrap() - available_memory().unwrap() / 4);
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
