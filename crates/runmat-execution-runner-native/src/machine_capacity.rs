//! Local host discovery, separated from portable numerical policies.
use std::io;

pub(crate) fn cpu_millicores() -> u32 {
    let threads = std::thread::available_parallelism()
        .map_or(1, usize::from)
        .min(runmat_cpu_execution::MAX_THREADS);
    threads as u32 * 1000
}

/// The memory this machine admits for work now (`runmat_cpu_execution::machine_memory`).
pub(crate) fn memory_bytes() -> io::Result<u64> {
    runmat_cpu_execution::machine_memory::admissible_memory_bytes()
}

/// Physical memory an analysis worker process holds before its first numerical store: the
/// executable's written data, the runtime, allocator metadata and thread stacks. Resource plans
/// charge numerical stores, which the worker's footprint holds on top of this. Measured
/// 2026-11-15 on a release worker as its footprint less its live heap at its first plan: 169 MB.
const WORKER_PROCESS_BASELINE_BYTES: u64 = 192 * 1024 * 1024;

/// The memory one process-isolated analysis worker of a local run may use for its plan: the run's
/// pool, less the memory this coordinating process holds now and the worker process's own
/// baseline. A plan admitted against this value is compared with the same physical memory the
/// pool was read from.
///
/// `pool_bytes` is the run's admitted capacity, read once when the run was configured
/// (`NativeExecutionConfig::local_memory_capacity_bytes`). The coordinator keeps its mesh, model
/// and prepared inputs resident while the worker solves, so the worker cannot also have them.
/// A run reads this once, after the family compile and before path selection, and uses the same
/// value for path selection, the output policy and the worker envelope.
pub(crate) fn worker_memory_bytes(pool_bytes: u64) -> io::Result<u64> {
    worker_budget(pool_bytes, process_footprint()?)
}

fn worker_budget(pool_bytes: u64, coordinator_bytes: u64) -> io::Result<u64> {
    match pool_bytes
        .checked_sub(coordinator_bytes)
        .and_then(|left| left.checked_sub(WORKER_PROCESS_BASELINE_BYTES))
    {
        Some(budget) if budget > 0 => Ok(budget),
        _ => Err(io::Error::other(format!(
            "the coordinating process holds {coordinator_bytes} bytes of the run's \
             {pool_bytes}-byte memory pool; with a worker process's own \
             {WORKER_PROCESS_BASELINE_BYTES} bytes no memory is left for its analysis"
        ))),
    }
}

/// Record this process's footprint at a coordinator phase boundary (info level).
pub(crate) fn trace_footprint(phase: &'static str) {
    if let Ok((bytes, peak)) =
        runmat_cpu_execution::machine_memory::process_footprint_and_peak_bytes()
    {
        tracing::info!(
            phase,
            footprint_bytes = bytes,
            peak_footprint_bytes = peak,
            "coordinator memory"
        );
    }
}

/// This process's physical footprint.
pub(crate) fn process_footprint() -> io::Result<u64> {
    runmat_cpu_execution::machine_memory::process_footprint_bytes()
}

/// Execution ceilings for one local meshing run: sized from the memory this host admits for work
/// (the same authority as publication staging and the thread plan, `run_pool`) and from the requested mesh
/// size. They bound whether the run completes, never what it produces. A host whose available
/// memory cannot be read gets an error; no portable constant stands in for its capacity.
pub fn meshing_execution_ceilings(
    limits: &runmat_meshing_core::MeshingSizeLimits,
) -> io::Result<runmat_meshing_core::MeshingExecutionCeilings> {
    Ok(meshing_execution_ceilings_for(memory_bytes()?, limits))
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

/// The memory this host admits for one run (`NativeExecutionConfig::local_memory_capacity_bytes`):
/// the configured pool when the run has one, else read now. The same value admits the run's solve
/// and sizes its thread plan, and becomes the authority for the run's study publication: a
/// publication may stage this pool less what this coordinating process holds when it reads its
/// budget. Study planning refuses a study whose planned publication exceeds that, before any
/// solve.
pub(crate) fn run_pool(configured: Option<u64>) -> io::Result<u64> {
    let pool = match configured {
        Some(pool) => pool,
        None => memory_bytes()?,
    };
    runmat_runtime::analysis::install_publication_memory_authority(std::sync::Arc::new(
        RunPublicationMemory { pool_bytes: pool },
    ))
    .map_err(io::Error::other)?;
    Ok(pool)
}

/// One run's admitted memory, read by the runtime each time a publication sizes its staging.
struct RunPublicationMemory {
    pool_bytes: u64,
}

impl runmat_runtime::analysis::PublicationMemoryAuthority for RunPublicationMemory {
    fn publication_memory(&self) -> Result<runmat_runtime::analysis::PublicationMemory, String> {
        Ok(runmat_runtime::analysis::PublicationMemory {
            admitted_bytes: self.pool_bytes,
            resident_bytes: process_footprint().map_err(|error| error.to_string())?,
        })
    }
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
    fn a_worker_gets_the_pool_less_what_the_coordinator_holds() {
        const GB: u64 = 1_000_000_000;
        assert_eq!(
            worker_budget(22 * GB, 3 * GB).unwrap(),
            19 * GB - WORKER_PROCESS_BASELINE_BYTES
        );
        assert!(worker_budget(3 * GB + WORKER_PROCESS_BASELINE_BYTES, 3 * GB).is_err());
        assert!(worker_budget(3 * GB, 3 * GB).is_err());
        assert!(worker_budget(2 * GB, 3 * GB).is_err());
        assert!(process_footprint().unwrap() > 0);
        let pool = memory_bytes().unwrap();
        assert!(worker_memory_bytes(pool).unwrap() < pool);
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
