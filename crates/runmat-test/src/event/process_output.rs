//! Raw native application streams belong to a worker, not an inferred test.
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::identity::WorkerSessionId;

/// Retained prefix per raw stream over one worker lifetime. Draining continues
/// after the bound; raw process output is distinct from per-attempt console text.
pub const MAX_WORKER_PROCESS_STREAM_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerProcessOutput {
    pub worker: WorkerSessionId,
    pub retirement: WorkerProcessRetirement,
    pub stdout: WorkerStreamOutput,
    pub stderr: WorkerStreamOutput,
}

impl WorkerProcessOutput {
    pub fn complete(&self) -> bool {
        self.retirement == WorkerProcessRetirement::Verified
            && self.stdout.completion == WorkerStreamCompletion::EndOfStream
            && self.stderr.completion == WorkerStreamCompletion::EndOfStream
    }

    pub fn validate(&self, maximum_stream_bytes: usize) -> Result<(), crate::TestDomainError> {
        if self.worker.0.is_empty() || self.worker.0.len() > 128 {
            return Err(crate::TestDomainError::InvalidField {
                field: "worker.process_output.worker",
                reason: "worker session identity is empty or exceeds its bound".into(),
            });
        }
        for stream in [&self.stdout, &self.stderr] {
            let invalid = stream.bytes.as_slice().len() > maximum_stream_bytes
                || stream
                    .observed_bytes
                    .is_some_and(|count| count < stream.bytes.as_slice().len() as u64)
                || (stream.observed_bytes.is_none() && !stream.truncated)
                || (stream.completion != WorkerStreamCompletion::Transferred
                    && stream.observed_bytes.is_some_and(|count| {
                        (count > stream.bytes.as_slice().len() as u64) != stream.truncated
                    }));
            if invalid {
                return Err(crate::TestDomainError::InvalidField {
                    field: "worker.process_output",
                    reason: "stream prefix exceeds its bound or contradicts its byte count".into(),
                });
            }
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

/// A bounded byte prefix. Bytes remain exact even when native output is not UTF-8.
/// `observed_bytes` counts the whole stream; None denotes counter overflow.
/// Late/background writes are never attributed to whichever test ran next.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WorkerStreamOutput {
    pub bytes: WorkerOutputBytes,
    pub observed_bytes: Option<u64>,
    pub truncated: bool,
    pub completion: WorkerStreamCompletion,
}

/// Immutable sharing across canonical events and report projections. Cloning
/// a report does not duplicate a worker's retained byte prefix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WorkerOutputBytes(Arc<Vec<u8>>);

impl WorkerOutputBytes {
    pub fn new(bytes: Vec<u8>) -> Self {
        Self(Arc::new(bytes))
    }
    pub fn as_slice(&self) -> &[u8] {
        self.0.as_slice()
    }
}

impl Serialize for WorkerOutputBytes {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WorkerOutputBytes {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Vec::<u8>::deserialize(deserializer).map(Self::new)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerStreamCompletion {
    NotCaptured,
    Reading,
    EndOfStream,
    ReadFailed { kind: String },
    Transferred,
}
