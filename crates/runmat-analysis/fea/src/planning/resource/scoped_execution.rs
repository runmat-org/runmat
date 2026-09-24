use std::{cell::Cell, fmt, panic::AssertUnwindSafe, thread};

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

/// Executes deterministic contiguous chunks in a fresh scoped thread set.
///
/// This function deliberately creates no global pool and performs no host CPU discovery. The
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
    let chunks = plan
        .chunks(item_count)
        .map_err(ScopedLocalExecutionError::InvalidPlan)?;
    let _guard = LocalChunkGuard::enter().ok_or(ScopedLocalExecutionError::NestedExecution)?;
    if chunks.len() == 1 {
        let chunk = chunks[0];
        return std::panic::catch_unwind(AssertUnwindSafe(|| work(chunk)))
            .map_err(|_| ScopedLocalExecutionError::WorkerPanicked { chunk })?
            .map(|result| vec![result])
            .map_err(|source| ScopedLocalExecutionError::Work { chunk, source });
    }

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let work = &work;
            let handle = thread::Builder::new().spawn_scoped(scope, move || {
                let _guard = LocalChunkGuard::enter().expect("new scoped worker is not active");
                work(chunk)
            });
            match handle {
                Ok(handle) => handles.push((chunk, handle)),
                Err(error) => {
                    for (_, handle) in handles {
                        let _ = handle.join();
                    }
                    return Err(ScopedLocalExecutionError::WorkerSpawnFailed {
                        chunk,
                        reason: error.to_string(),
                    });
                }
            }
        }
        let outcomes = handles
            .into_iter()
            .map(|(chunk, handle)| (chunk, handle.join()))
            .collect::<Vec<_>>();

        let mut results = Vec::with_capacity(outcomes.len());
        for (chunk, outcome) in outcomes {
            let result = outcome
                .map_err(|_| ScopedLocalExecutionError::WorkerPanicked { chunk })?
                .map_err(|source| ScopedLocalExecutionError::Work { chunk, source })?;
            results.push(result);
        }
        Ok(results)
    })
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
    let chunks = plan
        .chunks(values.len() as u64)
        .map_err(ScopedLocalExecutionError::InvalidPlan)?;
    let _guard = LocalChunkGuard::enter().ok_or(ScopedLocalExecutionError::NestedExecution)?;
    if chunks.len() == 1 {
        let chunk = chunks[0];
        return std::panic::catch_unwind(AssertUnwindSafe(|| work(chunk, values)))
            .map_err(|_| ScopedLocalExecutionError::WorkerPanicked { chunk })?
            .map_err(|source| ScopedLocalExecutionError::Work { chunk, source });
    }

    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(chunks.len());
        let mut remaining = values;
        for chunk in chunks {
            let len = (chunk.end_exclusive() - chunk.start()) as usize;
            let (slice, rest) = remaining.split_at_mut(len);
            remaining = rest;
            let work = &work;
            let handle = thread::Builder::new().spawn_scoped(scope, move || {
                let _guard = LocalChunkGuard::enter().expect("new scoped worker is not active");
                work(chunk, slice)
            });
            match handle {
                Ok(handle) => handles.push((chunk, handle)),
                Err(error) => {
                    for (_, handle) in handles {
                        let _ = handle.join();
                    }
                    return Err(ScopedLocalExecutionError::WorkerSpawnFailed {
                        chunk,
                        reason: error.to_string(),
                    });
                }
            }
        }
        let outcomes = handles
            .into_iter()
            .map(|(chunk, handle)| (chunk, handle.join()))
            .collect::<Vec<_>>();
        for (chunk, outcome) in outcomes {
            outcome
                .map_err(|_| ScopedLocalExecutionError::WorkerPanicked { chunk })?
                .map_err(|source| ScopedLocalExecutionError::Work { chunk, source })?;
        }
        Ok(())
    })
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
