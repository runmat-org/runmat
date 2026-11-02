//! Bounded original-attempt observations, independent of result payloads.
use runmat_execution::{identity::AttemptId, process_output::ProcessOutputBytes, JobHandle};
use serde::{Deserialize, Serialize};

use super::{LocalJobState, LocalJobWorkloadKind};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawProcessOutputCursor {
    pub stdout: u64,
    pub stderr: u64,
}

/// Host-observed process output, not a runtime console receipt or numerical result.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawProcessOutputPage {
    pub attempt_id: AttemptId,
    pub stdout: ProcessOutputBytes,
    pub stderr: ProcessOutputBytes,
    pub next: RawProcessOutputCursor,
    pub stdout_state: crate::RawOutputState,
    pub stderr_state: crate::RawOutputState,
    pub retired: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawProcessOutputAttempt {
    pub attempt_id: AttemptId,
    pub ordinal: u16,
    /// Logical driver closure only; neither pipe EOF nor physical retirement.
    pub terminal: Option<runmat_execution_runner::task::AttemptDisposition>,
    /// Original assignment did not start; preflight charges remain independent.
    pub not_started: bool,
}

/// Discover one ordinal without transporting arguments, results or full history.
/// An absent attempt is not proof that earlier attempts' output was delivered.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RawProcessOutputLookup {
    pub handle: JobHandle,
    pub task: Option<runmat_execution::TaskId>,
    pub ordinal: u32,
    pub state: LocalJobState,
    pub workload_kind: LocalJobWorkloadKind,
    pub retain_until_unix_millis: u64,
    pub attempt: Option<RawProcessOutputAttempt>,
}
