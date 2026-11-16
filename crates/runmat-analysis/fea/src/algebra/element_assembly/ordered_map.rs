//! Ordered parallel map over an index range.

/// `work(0..count)` in canonical order on the installed admitted pool (serial without one).
/// Each item is computed independently, so results never depend on the thread count.
pub fn ordered_map_range<R: Send>(count: usize, work: impl Fn(usize) -> R + Sync + Send) -> Vec<R> {
    const CHUNK: usize = 1024;
    if count <= CHUNK || runmat_cpu_execution::current_threads() == 1 {
        return (0..count).map(work).collect();
    }
    let ranges = (0..count.div_ceil(CHUNK))
        .map(|chunk| chunk * CHUNK..((chunk + 1) * CHUNK).min(count))
        .collect::<Vec<_>>();
    let usage = runmat_simulation_usage::UsageContext::capture();
    let progress = crate::progress::capture_fea_progress_context();
    let chunks = runmat_cpu_execution::ordered_map(&ranges, |range| {
        let _usage = usage
            .as_ref()
            .map(runmat_simulation_usage::UsageContext::enter);
        let _progress = crate::progress::replace_fea_progress_context(progress.clone());
        range.clone().map(&work).collect::<Vec<_>>()
    });
    let mut results = Vec::with_capacity(count);
    for chunk in chunks {
        results.extend(chunk);
    }
    results
}
