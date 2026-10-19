//! Project-scoped execution authority. Lease policy is shared with the harness;
//! filesystem providers own the clock, lock and durable record.
pub use runmat_workspace_contracts::execution::{
    execution_lease_is_current, transition_execution_lease, ExecutionFence, ExecutionLease,
    ExecutionLeaseCommand, ExecutionLeaseOutcome, ExecutionLeaseState, ExecutionOwnerId,
    ExecutionWorkId, SessionId, TurnId,
};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum FencedWorkspacePublicationOutcome {
    Publication {
        outcome: crate::workspace::WorkspacePublicationOutcome,
    },
    LeaseLost,
}

/// Retained ownership is bounded independently of source files. Authorities
/// must reclaim records through workflow retention before admitting more work.
pub const MAX_EXECUTION_RECORDS: usize = 10_000;

#[derive(Debug)]
pub(crate) struct LostLease;
impl std::fmt::Display for LostLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("execution ownership has expired or changed")
    }
}
impl std::error::Error for LostLease {}

pub(crate) fn require_current(
    state: Option<&ExecutionLeaseState>,
    lease: &ExecutionLease,
) -> std::io::Result<()> {
    if execution_lease_is_current(state, lease, authority_time_ms()?) {
        Ok(())
    } else {
        Err(std::io::Error::other(LostLease))
    }
}

pub(crate) fn fenced_outcome(
    result: std::io::Result<crate::workspace::WorkspacePublicationOutcome>,
) -> std::io::Result<FencedWorkspacePublicationOutcome> {
    match result {
        Ok(outcome) => Ok(FencedWorkspacePublicationOutcome::Publication { outcome }),
        Err(error) if error.get_ref().is_some_and(|error| error.is::<LostLease>()) => {
            Ok(FencedWorkspacePublicationOutcome::LeaseLost)
        }
        Err(error) => Err(error),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;

/// Wall-clock authority for lease expiry; `runmat_time` is safe on wasm hosts.
pub(crate) fn authority_time_ms() -> std::io::Result<i64> {
    i64::try_from(runmat_time::unix_timestamp_ms()).map_err(std::io::Error::other)
}

pub(crate) fn work_key(work: &ExecutionWorkId) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    // Length-delimited JSON of typed identity avoids ambiguous concatenation
    // and makes arbitrary user-supplied IDs safe as journal filenames.
    let bytes = serde_json::to_vec(work).map_err(std::io::Error::other)?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
