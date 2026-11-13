use super::*;
pub(super) mod groups;

/// Explicit configuration (including zero) overrides environment defaults.
pub(super) fn resolved_limits() -> Result<groups::Limits, String> {
    let policy = retention_config()
        .read()
        .map_err(|_| "retention lock poisoned")?
        .clone();
    let resolve = |configured: Option<usize>, primary: &str, fallback: &str| {
        configured.unwrap_or_else(|| {
            std::env::var(primary)
                .or_else(|_| std::env::var(fallback))
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(0)
        })
    };
    Ok(groups::Limits {
        total: resolve(
            policy.max_runs,
            "RUNMAT_FEA_ARTIFACT_MAX_RUNS",
            "RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS",
        ),
        per_kind: resolve(
            policy.max_runs_per_kind,
            "RUNMAT_FEA_ARTIFACT_MAX_RUNS_PER_KIND",
            "RUNMAT_ANALYSIS_ARTIFACT_MAX_RUNS_PER_KIND",
        ),
    })
}

pub(super) fn prune_filesystem_runs(
    store: &access::FilesystemAccess<'_>,
    run_id: &str,
) -> Result<(), AnalysisPublicationError> {
    publication::filesystem::transaction::prune_standalone(store, run_id)
}
