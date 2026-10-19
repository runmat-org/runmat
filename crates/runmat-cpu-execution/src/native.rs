use super::{CpuExecutionError, MAX_THREADS, WORKER_STACK_BYTES};
use rayon::prelude::*;
use std::{
    cell::Cell,
    sync::{Mutex, OnceLock, RwLock, TryLockError},
};

// A single resident pool per process prevents stacks multiplying across budgets.
// Owners with the same admitted capacity share it. Capacity changes wait for active
// readers before replacing the pool; nested kernels never allocate another pool.
thread_local! { static OWNING_POOL: Cell<bool> = const { Cell::new(false) }; }
// Rayon also supports foreign pools. Their workers have no admission from this
// provider and must never become implicit numerical execution capacity.
thread_local! { static ADMITTED_WORKER: Cell<bool> = const { Cell::new(false) }; }
struct Owner;
impl Owner {
    fn enter() -> Result<Self, CpuExecutionError> {
        OWNING_POOL.with(|active| {
            if active.replace(true) {
                Err(CpuExecutionError::NestedExecution)
            } else {
                Ok(Self)
            }
        })
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        OWNING_POOL.with(|active| active.set(false));
    }
}
#[derive(Default)]
struct PoolAllocation {
    resident: RwLock<Option<rayon::ThreadPool>>,
    transition: Mutex<()>,
}
static POOL: OnceLock<PoolAllocation> = OnceLock::new();

pub fn with_cpu_pool<T: Send>(
    threads: usize,
    work: impl FnOnce() -> T + Send,
) -> Result<T, CpuExecutionError> {
    with_cpu_pool_cancellable(threads, || false, work)
}

/// Acquires a compatible pool lease while observing host cancellation during capacity waits.
pub fn with_cpu_pool_cancellable<T: Send>(
    threads: usize,
    cancelled: impl Fn() -> bool,
    work: impl FnOnce() -> T + Send,
) -> Result<T, CpuExecutionError> {
    if threads == 0 || threads > MAX_THREADS {
        return Err(CpuExecutionError::InvalidThreadCount);
    }
    if rayon::current_thread_index().is_some() {
        return Err(CpuExecutionError::NestedExecution);
    }
    let _owner = Owner::enter()?;
    let allocation = POOL.get_or_init(PoolAllocation::default);
    loop {
        if cancelled() {
            return Err(CpuExecutionError::Cancelled);
        }
        let resident = match allocation.resident.try_read() {
            Ok(resident) => resident,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
            Err(TryLockError::WouldBlock) => {
                std::thread::park_timeout(std::time::Duration::from_millis(1));
                continue;
            }
        };
        let capacity = resident
            .as_ref()
            .map_or(1, |pool| pool.current_num_threads());
        if capacity == threads {
            // The read lease prevents replacement while any admitted owner uses this capacity.
            // One-thread work runs on its host worker, so independent serial batches overlap.
            return Ok(match resident.as_ref() {
                Some(pool) => pool.install(work),
                None => work(),
            });
        }
        drop(resident);
        // Serialize capacity transitions until the requesting owner acquires its read lease.
        // Otherwise different budgets could repeatedly replace one another before either runs.
        let transition = loop {
            if cancelled() {
                return Err(CpuExecutionError::Cancelled);
            }
            match allocation.transition.try_lock() {
                Ok(guard) => break guard,
                Err(TryLockError::Poisoned(error)) => break error.into_inner(),
                Err(TryLockError::WouldBlock) => {
                    std::thread::park_timeout(std::time::Duration::from_millis(1))
                }
            }
        };
        let mut resident = loop {
            if cancelled() {
                return Err(CpuExecutionError::Cancelled);
            }
            match allocation.resident.try_write() {
                Ok(guard) => break guard,
                Err(TryLockError::Poisoned(error)) => break error.into_inner(),
                Err(TryLockError::WouldBlock) => {
                    std::thread::park_timeout(std::time::Duration::from_millis(1))
                }
            }
        };
        if resident
            .as_ref()
            .map_or(1, |pool| pool.current_num_threads())
            != threads
        {
            // Release the former allocation before constructing a different admitted budget.
            *resident = None;
            if threads > 1 {
                *resident = Some(
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .stack_size(WORKER_STACK_BYTES as usize)
                        .thread_name(|index| format!("runmat-cpu-{index}"))
                        .start_handler(|_| ADMITTED_WORKER.with(|worker| worker.set(true)))
                        .exit_handler(|_| ADMITTED_WORKER.with(|worker| worker.set(false)))
                        .build()
                        .map_err(|error| CpuExecutionError::PoolCreation(error.to_string()))?,
                );
            }
        }
        drop(resident);
        // No other transition can acquire the write lock before this read lease is installed.
        let resident = allocation
            .resident
            .read()
            .unwrap_or_else(|error| error.into_inner());
        drop(transition);
        return Ok(match resident.as_ref() {
            Some(pool) => pool.install(work),
            None => work(),
        });
    }
}

pub fn current_threads() -> usize {
    if ADMITTED_WORKER.with(Cell::get) && rayon::current_thread_index().is_some() {
        rayon::current_num_threads()
    } else {
        1
    }
}

/// Indexed collection preserves input order, including errors represented in R.
/// Small batches execute on the caller to avoid queueing trivial work.
pub fn ordered_map<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    if current_threads() == 1 || items.len() < 2 {
        items.iter().map(work).collect()
    } else {
        items.par_iter().map(work).collect()
    }
}

/// Borrowed disjoint outputs; no model clone or full-sized intermediate result.
pub fn for_each_mut<T: Send>(items: &mut [T], work: impl Fn(&mut T) + Sync + Send) {
    if current_threads() == 1 || items.len() < 2 {
        items.iter_mut().for_each(work);
    } else {
        items.par_iter_mut().for_each(work);
    }
}

pub fn for_each_mut_indexed<T: Send>(items: &mut [T], work: impl Fn(usize, &mut T) + Sync + Send) {
    if current_threads() == 1 || items.len() < 2 {
        items
            .iter_mut()
            .enumerate()
            .for_each(|(index, value)| work(index, value));
    } else {
        items
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, value)| work(index, value));
    }
}

#[cfg(test)]
mod tests;
