//! Native transport for solver progress, shared by local and remote workers.
use std::sync::Arc;

use runmat_analysis_fea::{FeaCancellationToken, FeaProgressContext, FeaProgressContextGuard};
use runmat_execution::identity::AttemptId;

pub(crate) fn install(
    cancellation: FeaCancellationToken,
    attempt_id: Option<AttemptId>,
) -> FeaProgressContextGuard {
    let recorded_attempt = attempt_id.map(|id| id.to_string());
    runmat_analysis_fea::replace_fea_progress_context(FeaProgressContext {
        handler: Some(Arc::new(move |event| {
            tracing::info!(
                event_code = "analysis.solver.progress",
                attempt_id = recorded_attempt.as_deref().unwrap_or(""),
                operation = %event.operation,
                phase = ?event.phase,
                status = ?event.status,
                current = ?event.current,
                "analysis solver progress: {}",
                event.message
            );
        })),
        cancellation,
        attempt_id,
    })
}

#[cfg(test)]
mod tests;
