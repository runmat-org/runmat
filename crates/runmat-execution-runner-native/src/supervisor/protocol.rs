use runmat_execution::JobId;
use serde::{Deserialize, Serialize};

use super::{
    BatchSubmission, GraphBatchSubmission, JobAttachment, LocalJobRecord, ProgramBatchSubmission,
};

pub const SUPERVISOR_PROTOCOL_VERSION: u16 = 7;
pub const SUPERVISOR_MAX_MESSAGE_BYTES: u32 = 64 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupervisorRequest {
    #[serde(default)]
    pub local_fea_organization: runmat_runtime::analysis::fea_run_admission::FeaLocalOrganization,
    pub protocol_version: u16,
    pub authentication_token: String,
    pub config_identity: runmat_execution::Digest,
    pub command: SupervisorCommand,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "command", deny_unknown_fields)]
pub enum SupervisorCommand {
    Ping,
    ExecutionContext {
        idempotency_key: Option<String>,
    },
    Submit {
        submission: Box<BatchSubmission>,
    },
    SubmitProgram {
        submission: Box<ProgramBatchSubmission>,
    },
    SubmitGraph {
        submission: Box<GraphBatchSubmission>,
    },
    List,
    Show {
        job_id: JobId,
    },
    Attach {
        job_id: JobId,
        stdout_offset: u64,
        stderr_offset: u64,
    },
    ConsolePage {
        handle: runmat_execution::JobHandle,
        attempt_id: runmat_execution::identity::AttemptId,
        cursor: runmat_execution::console::ConsoleCursor,
        budget: runmat_execution::console::ConsoleReadBudget,
    },
    RawOutputPage {
        handle: runmat_execution::JobHandle,
        attempt_id: runmat_execution::identity::AttemptId,
        cursor: super::RawProcessOutputCursor,
    },
    RawOutputAttempt {
        handle: runmat_execution::JobHandle,
        ordinal: u32,
    },
    Cancel {
        job_id: JobId,
    },
    CancelHandle {
        handle: runmat_execution::JobHandle,
        reason: runmat_execution::CancellationReason,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "response", deny_unknown_fields)]
pub enum SupervisorResponse {
    Pong {
        config_identity: runmat_execution::Digest,
    },
    ExecutionContext {
        context: super::DurableExecutionContext,
    },
    Submitted {
        record: LocalJobRecord,
        duplicate: bool,
    },
    Jobs {
        records: Vec<LocalJobRecord>,
    },
    Job {
        record: LocalJobRecord,
    },
    Attachment {
        attachment: Box<JobAttachment>,
    },
    ConsolePage {
        attempt_id: runmat_execution::identity::AttemptId,
        page: Option<Box<runmat_execution::console::ConsolePage>>,
    },
    RawOutputPage {
        attempt_id: runmat_execution::identity::AttemptId,
        page: Option<Box<super::RawProcessOutputPage>>,
    },
    RawOutputAttempt {
        lookup: super::RawProcessOutputLookup,
    },
    Cancelled {
        record: LocalJobRecord,
    },
    Error {
        code: String,
        message: String,
    },
}

impl SupervisorResponse {
    pub(super) fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Error {
            code: code.into(),
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_control_requires_the_original_reason_and_complete_handle() {
        let handle = runmat_execution::JobHandle {
            id: JobId::derive(&[b"control-job"]),
            run_id: runmat_execution::RunId::derive(&[b"control-run"]),
            generation: 1,
            outputs: runmat_execution::OutputContract {
                requested_outputs: 2,
                owned_work: None,
            },
        };
        let original = SupervisorCommand::CancelHandle {
            handle: handle.clone(),
            reason: runmat_execution::CancellationReason::Quota,
        };
        let value = serde_json::to_value(original).unwrap();
        assert!(
            matches!(serde_json::from_value::<SupervisorCommand>(value.clone()).unwrap(),
            SupervisorCommand::CancelHandle { handle: retained, reason: runmat_execution::CancellationReason::Quota } if retained == handle)
        );
        for field in ["reason", "handle"] {
            let mut missing = value.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<SupervisorCommand>(missing).is_err());
        }
        let mut incomplete = value;
        incomplete["handle"]
            .as_object_mut()
            .unwrap()
            .remove("generation");
        assert!(serde_json::from_value::<SupervisorCommand>(incomplete).is_err());
    }
}
