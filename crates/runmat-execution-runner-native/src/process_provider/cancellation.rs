//! Deliver an existing scheduler cause with its complete fenced attempt.
use runmat_execution::CancellationReason;
use runmat_execution_runner::{AttemptRequest, TaskCancellationCause};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CancellationCommand {
    pub(crate) request: AttemptRequest,
    pub(crate) cause: TaskCancellationCause,
}

/// A provider report is an observation, not authority to manufacture a user
/// stop. Transport faults likewise use failure/retirement, not this conversion.
pub(crate) fn reason(
    request: &AttemptRequest,
    cause: TaskCancellationCause,
) -> Result<CancellationReason, String> {
    cause.reason_for_attempt(request)
}
