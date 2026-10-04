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

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug)]
pub(crate) struct LostLease;
#[cfg(not(target_arch = "wasm32"))]
impl std::fmt::Display for LostLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("execution ownership has expired or changed")
    }
}
#[cfg(not(target_arch = "wasm32"))]
impl std::error::Error for LostLease {}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn authority_time_ms() -> std::io::Result<i64> {
    let elapsed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(std::io::Error::other)?;
    i64::try_from(elapsed.as_millis()).map_err(std::io::Error::other)
}

#[cfg(not(target_arch = "wasm32"))]
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
