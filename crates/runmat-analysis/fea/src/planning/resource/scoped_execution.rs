use std::{cell::Cell, fmt, panic::AssertUnwindSafe};

use super::{LocalExecutionPlan, LocalExecutionPlanError, LocalWorkChunk};

/// Failure returned by one scoped local execution.
///
/// Work failures are reported in canonical chunk order even though workers may finish in a
/// different order. A worker panic is contained at this boundary and never unwinds through the
/// numerical caller.
#[derive(Debug)]
pub enum ScopedLocalExecutionError<E> {
    InvalidPlan(LocalExecutionPlanError),
    NestedExecution,
    Work {
        chunk: LocalWorkChunk,
        source: E,
    },
    WorkerPanicked {
        chunk: LocalWorkChunk,
    },
    WorkerSpawnFailed {
        chunk: LocalWorkChunk,
        reason: String,
    },
}

impl<E: fmt::Display> fmt::Display for ScopedLocalExecutionError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(error) => write!(formatter, "{error}"),
            Self::NestedExecution => {
                write!(formatter, "nested local numerical execution is forbidden")
            }
            Self::Work { chunk, source } => write!(
                formatter,
                "local execution chunk {} failed: {source}",
                chunk.ordinal()
            ),
            Self::WorkerPanicked { chunk } => write!(
                formatter,
                "local execution chunk {} panicked",
                chunk.ordinal()
            ),
            Self::WorkerSpawnFailed { chunk, reason } => write!(
                formatter,
                "could not start local execution chunk {}: {reason}",
                chunk.ordinal()
            ),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for ScopedLocalExecutionError<E> {}

thread_local! {
    static EXECUTING_LOCAL_CHUNK: Cell<bool> = const { Cell::new(false) };
}

/// True while this thread executes a chunk of an admitted local plan. Work that would open its
/// own parallel execution must fail closed instead of oversubscribing the admitted cores.
pub(crate) fn local_chunk_active() -> bool {
    EXECUTING_LOCAL_CHUNK.with(Cell::get)
}

struct LocalChunkGuard;

impl LocalChunkGuard {
    fn enter() -> Option<Self> {
        EXECUTING_LOCAL_CHUNK.with(|active| {
            if active.get() {
                None
            } else {
                active.set(true);
                Some(Self)
            }
        })
    }
}

impl Drop for LocalChunkGuard {
    fn drop(&mut self) {
        EXECUTING_LOCAL_CHUNK.with(|active| active.set(false));
    }
}

fn execute_in_pool<T: Send>(
    plan: LocalExecutionPlan,
    work: impl FnOnce() -> T + Send,
) -> Result<T, runmat_cpu_execution::CpuExecutionError> {
    let threads = usize::from(plan.thread_count().get());
    if runmat_cpu_execution::current_threads() > 1 {
        Ok(work())
    } else {
        runmat_cpu_execution::with_cpu_pool(threads, work)
    }
}

/// The admitted pool a kernel installed for its whole step is the execution authority for
/// every nested scoped operation; otherwise the caller's plan is. Scoped operations are
/// required to be partition independent (fixed reduction blocks, canonical chunk order), so
/// this choice never changes numerical results.
fn effective_plan(plan: LocalExecutionPlan) -> LocalExecutionPlan {
    let installed = runmat_cpu_execution::current_threads();
    if installed > 1 {
        u16::try_from(installed)
            .ok()
            .and_then(|threads| super::LocalThreadCount::new(threads).ok())
            .and_then(|threads| LocalExecutionPlan::fixed(threads).ok())
            .unwrap_or(plan)
    } else {
        plan
    }
}

/// Executes deterministic contiguous chunks in the admitted reusable CPU pool.
///
/// This function performs no host CPU discovery. The process retains one bounded pool; the
/// caller-supplied plan is the complete concurrency authority. Results and failures are returned
/// in canonical chunk order, independently of worker completion order.
pub fn execute_scoped_chunks<T, E, F>(
    plan: LocalExecutionPlan,
    item_count: u64,
    work: F,
) -> Result<Vec<T>, ScopedLocalExecutionError<E>>
where
    T: Send,
    E: Send,
    F: Fn(LocalWorkChunk) -> Result<T, E> + Sync,
{
    let plan = effective_plan(plan);
    let chunks = plan
        .chunks(item_count)
        .map_err(ScopedLocalExecutionError::InvalidPlan)?;
    if EXECUTING_LOCAL_CHUNK.with(Cell::get) {
        return Err(ScopedLocalExecutionError::NestedExecution);
    }
    if chunks.len() == 1 {
        let _guard = LocalChunkGuard::enter().expect("checked serial execution");
        let chunk = chunks[0];
        return std::panic::catch_unwind(AssertUnwindSafe(|| work(chunk)))
            .map_err(|_| ScopedLocalExecutionError::WorkerPanicked { chunk })?
            .map(|result| vec![result])
            .map_err(|source| ScopedLocalExecutionError::Work { chunk, source });
    }

    let progress = crate::progress::capture_fea_progress_context();
    execute_in_pool(plan, || {
        runmat_cpu_execution::ordered_map(&chunks, |&chunk| {
            let _guard = LocalChunkGuard::enter().expect("pool worker is not active");
            let _progress = crate::progress::replace_fea_progress_context(progress.clone());
            (
                chunk,
                std::panic::catch_unwind(AssertUnwindSafe(|| work(chunk))),
            )
        })
    })
    .map_err(|error| ScopedLocalExecutionError::WorkerSpawnFailed {
        chunk: chunks[0],
        reason: error.to_string(),
    })?
    .into_iter()
    .map(|(chunk, outcome)| {
        outcome
            .map_err(|_| ScopedLocalExecutionError::WorkerPanicked { chunk })?
            .map_err(|source| ScopedLocalExecutionError::Work { chunk, source })
    })
    .collect()
}

/// Updates disjoint contiguous slices under the admitted local plan without a full-size
/// temporary output. The caller owns the buffer; each worker receives only its canonical slice.
pub fn execute_scoped_slices_mut<T, E, F>(
    plan: LocalExecutionPlan,
    values: &mut [T],
    work: F,
) -> Result<(), ScopedLocalExecutionError<E>>
where
    T: Send,
    E: Send,
    F: Fn(LocalWorkChunk, &mut [T]) -> Result<(), E> + Sync,
{
    if values.is_empty() {
        return Ok(());
    }
    let plan = effective_plan(plan);
    let chunks = plan
        .chunks(values.len() as u64)
        .map_err(ScopedLocalExecutionError::InvalidPlan)?;
    if EXECUTING_LOCAL_CHUNK.with(Cell::get) {
        return Err(ScopedLocalExecutionError::NestedExecution);
    }
    if chunks.len() == 1 {
        let _guard = LocalChunkGuard::enter().expect("checked serial execution");
        let chunk = chunks[0];
        return std::panic::catch_unwind(AssertUnwindSafe(|| work(chunk, values)))
            .map_err(|_| ScopedLocalExecutionError::WorkerPanicked { chunk })?
            .map_err(|source| ScopedLocalExecutionError::Work { chunk, source });
    }

    let mut remaining = values;
    let mut slices = Vec::with_capacity(chunks.len());
    for &chunk in &chunks {
        let (slice, rest) = remaining.split_at_mut(chunk.len() as usize);
        remaining = rest;
        slices.push((chunk, slice, None));
    }
    let progress = crate::progress::capture_fea_progress_context();
    execute_in_pool(plan, || {
        runmat_cpu_execution::for_each_mut(&mut slices, |(chunk, slice, outcome)| {
            let _guard = LocalChunkGuard::enter().expect("pool worker is not active");
            let _progress = crate::progress::replace_fea_progress_context(progress.clone());
            *outcome = Some(std::panic::catch_unwind(AssertUnwindSafe(|| {
                work(*chunk, slice)
            })));
        });
    })
    .map_err(|error| ScopedLocalExecutionError::WorkerSpawnFailed {
        chunk: chunks[0],
        reason: error.to_string(),
    })?;
    for (chunk, _, outcome) in slices {
        outcome
            .expect("joined worker outcome")
            .map_err(|_| ScopedLocalExecutionError::WorkerPanicked { chunk })?
            .map_err(|source| ScopedLocalExecutionError::Work { chunk, source })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planning::resource::LocalThreadCount;

    fn fixed(count: u16) -> LocalExecutionPlan {
        LocalExecutionPlan::fixed(LocalThreadCount::new(count).unwrap()).unwrap()
    }

    #[test]
    fn scoped_results_are_returned_in_canonical_chunk_order() {
        let results = execute_scoped_chunks(fixed(3), 8, |chunk| {
            if chunk.ordinal() == 0 {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Ok::<_, &'static str>((chunk.ordinal(), chunk.start(), chunk.end_exclusive()))
        })
        .unwrap();
        assert_eq!(results, vec![(0, 0, 3), (1, 3, 6), (2, 6, 8)]);
    }

    #[test]
    fn scoped_failures_use_the_lowest_canonical_chunk() {
        let error = execute_scoped_chunks(fixed(3), 3, |chunk| {
            if chunk.ordinal() == 0 {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err::<(), _>(format!("failure-{}", chunk.ordinal()))
        })
        .unwrap_err();
        match error {
            ScopedLocalExecutionError::Work { chunk, source } => {
                assert_eq!(chunk.ordinal(), 0);
                assert_eq!(source, "failure-0");
            }
            other => panic!("unexpected scoped failure: {other:?}"),
        }
    }

    #[test]
    fn worker_panics_are_contained() {
        let error = execute_scoped_chunks(fixed(2), 2, |chunk| {
            if chunk.ordinal() == 0 {
                panic!("contained test panic");
            }
            Ok::<_, ()>(chunk.ordinal())
        })
        .unwrap_err();
        assert!(matches!(
            error,
            ScopedLocalExecutionError::WorkerPanicked { chunk } if chunk.ordinal() == 0
        ));
    }

    #[test]
    fn serial_panics_are_contained_and_guard_is_reset() {
        let error = execute_scoped_chunks(fixed(2), 1, |_chunk| -> Result<(), ()> {
            panic!("contained serial panic");
        })
        .unwrap_err();
        assert!(matches!(
            error,
            ScopedLocalExecutionError::WorkerPanicked { .. }
        ));
        assert_eq!(
            execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| Ok::<_, ()>(1)).unwrap(),
            vec![1],
        );
    }

    #[test]
    fn nested_execution_is_rejected_in_workers() {
        let results = execute_scoped_chunks(fixed(2), 2, |_| {
            Ok::<_, ()>(execute_scoped_chunks(fixed(2), 2, |_| Ok::<_, ()>(1)))
        })
        .unwrap();
        assert!(results
            .iter()
            .all(|result| matches!(result, Err(ScopedLocalExecutionError::NestedExecution))));
    }

    #[test]
    fn mutable_slices_cover_canonical_intervals_without_extra_output() {
        let mut values = [0_u64; 8];
        execute_scoped_slices_mut(fixed(3), &mut values, |chunk, slice| {
            for (offset, value) in slice.iter_mut().enumerate() {
                *value = chunk.start() + offset as u64;
            }
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(values, [0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn mutable_slice_failures_use_canonical_order_and_release_the_guard() {
        let mut values = [0_u8; 3];
        let error = execute_scoped_slices_mut(fixed(3), &mut values, |chunk, _| {
            if chunk.ordinal() == 0 {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err::<(), _>(chunk.ordinal())
        })
        .unwrap_err();
        assert!(matches!(
            error,
            ScopedLocalExecutionError::Work { chunk, source: 0 } if chunk.ordinal() == 0
        ));
        execute_scoped_slices_mut(fixed(3), &mut values, |_, slice| {
            slice.fill(1);
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(values, [1; 3]);
    }

    #[test]
    fn mutable_slices_reject_nested_execution() {
        let nested = execute_scoped_chunks(LocalExecutionPlan::serial(), 1, |_| {
            let mut values = [0_u8; 2];
            Ok::<_, ()>(execute_scoped_slices_mut(fixed(2), &mut values, |_, _| {
                Ok::<_, ()>(())
            }))
        })
        .unwrap();
        assert!(matches!(
            nested[0],
            Err(ScopedLocalExecutionError::NestedExecution)
        ));
    }
}

#[cfg(test)]
mod installed_pool_tests {
    use super::*;
    use crate::planning::resource::LocalThreadCount;
    #[test]
    fn admitted_outer_pool_is_reused_and_chunk_nesting_stays_rejected() {
        let plan = LocalExecutionPlan::fixed(LocalThreadCount::new(4).unwrap()).unwrap();
        let cancellation = crate::progress::FeaCancellationToken::new();
        cancellation.cancel();
        let _context =
            crate::progress::replace_fea_progress_context(crate::progress::FeaProgressContext {
                cancellation,
                ..Default::default()
            });
        let context = crate::progress::capture_fea_progress_context();
        let results = runmat_cpu_execution::with_cpu_pool(4, || {
            let _context = crate::progress::replace_fea_progress_context(context);
            execute_scoped_chunks(plan, 32, |chunk| {
                assert!(crate::progress::is_cancelled());
                assert!(matches!(
                    execute_scoped_chunks(plan, 32, |_| Ok::<_, ()>(())),
                    Err(ScopedLocalExecutionError::NestedExecution)
                ));
                Ok::<_, ()>(chunk.ordinal())
            })
        })
        .unwrap()
        .unwrap();
        assert_eq!(results, vec![0, 1, 2, 3]);
    }

    #[test]
    fn installed_pool_is_the_authority_for_nested_serial_plans() {
        let chunks = runmat_cpu_execution::with_cpu_pool(3, || {
            execute_scoped_chunks(LocalExecutionPlan::serial(), 9, |chunk| {
                Ok::<_, ()>((chunk.start(), chunk.end_exclusive()))
            })
        })
        .unwrap()
        .unwrap();
        assert_eq!(chunks, vec![(0, 3), (3, 6), (6, 9)]);
        let serial = execute_scoped_chunks(LocalExecutionPlan::serial(), 9, |chunk| {
            Ok::<_, ()>(chunk.len())
        })
        .unwrap();
        assert_eq!(serial, vec![9]);
    }
}
