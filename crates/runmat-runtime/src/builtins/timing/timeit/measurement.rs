//! Timeit sampling through the invocation's monotonic timing capability.

use super::{
    TimeitCallable, LOOP_COUNT_LIMIT, MAX_BATCH_SECONDS, MAX_SAMPLE_COUNT, MIN_SAMPLE_COUNT,
    TARGET_BATCH_SECONDS,
};
use crate::context::RuntimeContext;
use std::cmp::Ordering;

pub(super) async fn determine_loop_count(
    callable: &TimeitCallable,
    runtime: &RuntimeContext,
) -> Result<usize, crate::RuntimeError> {
    let mut loops = 1usize;
    loop {
        let elapsed = run_batch(callable, runtime, loops).await?;
        if elapsed >= TARGET_BATCH_SECONDS
            || elapsed >= MAX_BATCH_SECONDS
            || loops >= LOOP_COUNT_LIMIT
        {
            return Ok(loops);
        }
        loops = loops.saturating_mul(2);
        if loops == 0 {
            return Ok(LOOP_COUNT_LIMIT);
        }
    }
}

pub(super) async fn collect_samples(
    callable: &TimeitCallable,
    runtime: &RuntimeContext,
    loop_count: usize,
) -> Result<Vec<f64>, crate::RuntimeError> {
    let mut samples = Vec::with_capacity(MIN_SAMPLE_COUNT);
    while samples.len() < MIN_SAMPLE_COUNT {
        let elapsed = run_batch(callable, runtime, loop_count).await?;
        let per_iter = elapsed / loop_count as f64;
        samples.push(per_iter);
        if samples.len() >= MAX_SAMPLE_COUNT || elapsed >= MAX_BATCH_SECONDS {
            break;
        }
    }
    Ok(samples)
}

pub(super) async fn run_batch(
    callable: &TimeitCallable,
    runtime: &RuntimeContext,
    loop_count: usize,
) -> Result<f64, crate::RuntimeError> {
    let timing = runtime.service_ports().timing()?;
    let start = timing.monotonic_elapsed();
    let mut cooperation = runtime.cpu_cooperation();
    for _ in 0..loop_count {
        // The shared execution owner cancels a pending callback by dropping its
        // future. A completed invocation is never replayed on resumption.
        cooperation
            .await_cancellable(callable.invoke(runtime))
            .await
            .map_err(crate::execution::ExecutionServiceError::into_runtime_error)??;
        cooperation
            .checkpoint(1)
            .await
            .map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
    }
    let end = timing.monotonic_elapsed();
    Ok(end.saturating_sub(start).as_secs_f64())
}

pub(super) fn compute_median(mut samples: Vec<f64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(b).unwrap_or_else(|| {
            if a < b {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        }),
    });
    let mid = samples.len() / 2;
    if samples.len() % 2 == 1 {
        samples[mid]
    } else {
        (samples[mid - 1] + samples[mid]) * 0.5
    }
}
