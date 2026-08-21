use runmat_execution::JobId;
use serde::{Deserialize, Serialize};

use super::{BatchSubmission, JobAttachment, LocalJobRecord, ProgramBatchSubmission};

pub const SUPERVISOR_PROTOCOL_VERSION: u16 = 1;
pub const SUPERVISOR_MAX_MESSAGE_BYTES: u32 = 64 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupervisorRequest {
    pub protocol_version: u16,
    pub authentication_token: String,
    pub command: SupervisorCommand,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "command", deny_unknown_fields)]
pub enum SupervisorCommand {
    Ping,
    Submit {
        submission: Box<BatchSubmission>,
    },
    SubmitProgram {
        submission: Box<ProgramBatchSubmission>,
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
    Cancel {
        job_id: JobId,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "response", deny_unknown_fields)]
pub enum SupervisorResponse {
    Pong,
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
        attachment: JobAttachment,
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
