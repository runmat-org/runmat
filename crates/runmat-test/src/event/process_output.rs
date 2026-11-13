//! Raw native application streams belong to a worker, not an inferred test.
use runmat_execution::process_output::{ProcessStreamCompletion, ProcessStreamOutput};
use serde::{Deserialize, Serialize};

use crate::identity::WorkerSessionId;

/// Retained prefix per raw stream over one worker lifetime. Draining continues
/// after the bound; raw process output is distinct from per-attempt console text.
pub const MAX_WORKER_PROCESS_STREAM_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerProcessOutput {
    pub worker: WorkerSessionId,
    pub retirement: WorkerProcessRetirement,
    pub stdout: ProcessStreamOutput,
    pub stderr: ProcessStreamOutput,
}

impl WorkerProcessOutput {
    pub fn complete(&self) -> bool {
        self.retirement == WorkerProcessRetirement::Verified
            && self.stdout.completion == ProcessStreamCompletion::EndOfStream
            && self.stderr.completion == ProcessStreamCompletion::EndOfStream
    }

    pub fn validate(&self, maximum_stream_bytes: usize) -> Result<(), crate::TestDomainError> {
        if self.worker.0.is_empty() || self.worker.0.len() > 128 {
            return Err(crate::TestDomainError::InvalidField {
                field: "worker.process_output.worker",
                reason: "worker session identity is empty or exceeds its bound".into(),
            });
        }
        for stream in [&self.stdout, &self.stderr] {
            stream.validate(maximum_stream_bytes).map_err(|error| {
                crate::TestDomainError::InvalidField {
                    field: "worker.process_output",
                    reason: error.to_string(),
                }
            })?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerProcessRetirement {
    Verified,
    Unverified,
}
