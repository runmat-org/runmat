//! Shared CPU mechanics. Domains own partitioning, resource admission and result semantics.
//! No implicit global Rayon pool: callers must install an explicit admitted budget.
/// Actual package build configuration used for internal artifact reuse.
#[doc(hidden)]
pub const BUILD_CONFIGURATION: &str = env!("RUNMAT_BUILD_CONFIGURATION");
mod cooperative_sort;
pub use cooperative_sort::sort_copy_with_checkpoint;
mod in_place_sort;
pub use in_place_sort::sort_in_place_with_checkpoint;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{
    current_threads, for_each_mut, for_each_mut_indexed, ordered_map, with_cpu_pool,
    with_cpu_pool_cancellable,
};

/// Reserved stack per persistent worker; numerical scratch is additional.
pub const WORKER_STACK_BYTES: u64 = 2 * 1024 * 1024;
pub const MAX_THREADS: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CpuExecutionError {
    InvalidThreadCount,
    NestedExecution,
    Cancelled,
    PoolCreation(String),
}
impl std::fmt::Display for CpuExecutionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidThreadCount => formatter.write_str("CPU thread count is outside 1..=1024"),
            Self::NestedExecution => {
                formatter.write_str("nested CPU pool installation is forbidden")
            }
            Self::Cancelled => formatter.write_str("CPU pool acquisition cancelled"),
            Self::PoolCreation(reason) => {
                write!(formatter, "CPU worker pool creation failed: {reason}")
            }
        }
    }
}
impl std::error::Error for CpuExecutionError {}

#[cfg(target_arch = "wasm32")]
pub fn with_cpu_pool<T: Send>(
    threads: usize,
    work: impl FnOnce() -> T + Send,
) -> Result<T, CpuExecutionError> {
    if threads != 1 {
        return Err(CpuExecutionError::InvalidThreadCount);
    }
    Ok(work())
}
#[cfg(target_arch = "wasm32")]
pub fn with_cpu_pool_cancellable<T: Send>(
    threads: usize,
    cancelled: impl Fn() -> bool,
    work: impl FnOnce() -> T + Send,
) -> Result<T, CpuExecutionError> {
    if threads != 1 {
        return Err(CpuExecutionError::InvalidThreadCount);
    }
    if cancelled() {
        return Err(CpuExecutionError::Cancelled);
    }
    Ok(work())
}
#[cfg(target_arch = "wasm32")]
pub fn ordered_map<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync + Send) -> Vec<R> {
    items.iter().map(work).collect()
}
#[cfg(target_arch = "wasm32")]
pub fn for_each_mut<T: Send>(items: &mut [T], work: impl Fn(&mut T) + Sync + Send) {
    items.iter_mut().for_each(work);
}
#[cfg(target_arch = "wasm32")]
pub fn current_threads() -> usize {
    1
}

#[cfg(target_arch = "wasm32")]
pub fn for_each_mut_indexed<T: Send>(items: &mut [T], work: impl Fn(usize, &mut T) + Sync + Send) {
    items
        .iter_mut()
        .enumerate()
        .for_each(|(index, value)| work(index, value));
}
