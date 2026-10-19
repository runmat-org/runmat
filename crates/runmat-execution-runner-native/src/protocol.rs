use serde::{Deserialize, Serialize};

pub use runmat_execution_artifact::{
    ProgramExecutionResponse as WorkerResponse, ReservedProgramExecutionRequest,
    WorkerProgramRequest as WorkerRequest, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};

/// The Direct arm serializes as the existing WorkerRequest with no new fields.
/// A Graph child receives only a reference to its parent's owned artifact
/// store; it must verify and deserialize those exact bytes before effects.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum WorkerLaunchRequest {
    Direct(Box<WorkerRequest>),
    /// A Direct program with the scheduler's reserved resources, so the worker
    /// admits CPU parallelism within its own reservation.
    Reserved(Box<ReservedProgramExecutionRequest>),
    StagedGraph(Box<StagedGraphWorkerLaunch>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StagedGraphWorkerLaunch {
    pub staged_graph_v10: StagedGraphWorkerReference,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StagedGraphWorkerReference {
    pub artifact_id: runmat_execution::identity::ArtifactId,
    pub stored_digest: runmat_execution::Digest,
    pub stored_length: u64,
    pub assignment: runmat_execution::ProgramExecutionAssignment,
    pub context: runmat_execution::ProgramInvocationContext,
    pub requested_outputs: u16,
}

pub const NATIVE_WORKER_MESSAGE_SCHEMA_V2: u16 = 2;
const MAX_PROGRESS_PAYLOAD_BYTES: usize = 1024 * 1024;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramProgress {
    pub schema_version: u16,
    pub sequence: u64,
    pub media_type: String,
    pub value_schema: String,
    pub payload: Vec<u8>,
}

impl ProgramProgress {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != NATIVE_WORKER_MESSAGE_SCHEMA_V2
            || self.sequence == 0
            || self.media_type.is_empty()
            || self.media_type.len() > 128
            || !self.media_type.is_ascii()
            || self.media_type.chars().any(char::is_whitespace)
            || self.value_schema.is_empty()
            || self.value_schema.len() > 128
            || !self.value_schema.is_ascii()
            || self.value_schema.chars().any(char::is_whitespace)
            || self.payload.is_empty()
            || self.payload.len() > MAX_PROGRESS_PAYLOAD_BYTES
        {
            return Err("native program progress is malformed or exceeds its bound".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "message", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerProcessMessage {
    Progress {
        progress: ProgramProgress,
    },
    Console {
        record: runmat_execution::console::ConsoleRecord,
    },
    CollectiveRequest {
        request: runmat_execution::CollectiveRequest,
    },
    Completed {
        response: WorkerResponse,
        #[serde(deserialize_with = "required_option")]
        console: Option<runmat_execution::console::ConsoleReceipt>,
    },
    /// Worker-side admission/transport failure before a Program response
    /// exists. It cannot be mistaken for a user-code execution failure.
    InfrastructureFailure {
        #[serde(rename = "detail")]
        message: String,
    },
    /// The selected foreign host could not retire gracefully. The child
    /// retains its host and private roots until the parent kills and reaps
    /// this one-shot worker; this is never a user-code Graph failure.
    RetirementUnverified {
        #[serde(rename = "detail")]
        message: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum CollectiveProcessResult {
    Completed {
        response: runmat_execution::CollectiveResponse,
    },
    Failed {
        message: String,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "message", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerDriverMessage {
    CollectiveCompletion {
        context: runmat_execution::SpmdTaskContext,
        id: runmat_types::CollectiveId,
        sequence: runmat_execution::CollectiveSequence,
        result: CollectiveProcessResult,
    },
    /// Graph-only cooperative cancellation for the exact running attempt.
    /// Direct keeps its previous immediate process-retirement behavior.
    CancelAttempt {
        attempt_id: runmat_execution::identity::AttemptId,
        reason: runmat_execution::CancellationReason,
    },
}

/// The Direct variant keeps the existing stored shape for admitted tasks.
/// A Graph task owns its one payload in the artifact store; scheduler inputs
/// remain empty and must never become a second numerical payload authority.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StoredProgram {
    Direct(Box<StoredDirectProgram>),
    Graph(Box<StoredGraphProgram>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredDirectProgram {
    pub recipe: runmat_execution_artifact::ProgramBuildRecipe,
    pub artifact: runmat_execution_artifact::ProgramArtifact,
}

fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    serde::Deserialize::deserialize(deserializer)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredGraphProgram {
    pub request: runmat_execution_artifact::GraphExecutionRequest,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_envelope_is_bounded_and_strictly_shaped() {
        let progress = ProgramProgress {
            schema_version: NATIVE_WORKER_MESSAGE_SCHEMA_V2,
            sequence: 1,
            media_type: "application/vnd.runmat.progress+cbor".into(),
            value_schema: "runmat.progress.v1".into(),
            payload: vec![1],
        };
        progress.validate().unwrap();

        let mut empty = progress.clone();
        empty.payload.clear();
        assert!(empty.validate().is_err());
        let mut oversized = progress;
        oversized.payload = vec![0; MAX_PROGRESS_PAYLOAD_BYTES + 1];
        assert!(oversized.validate().is_err());

        let unknown = br#"{"message":"progress","progress":{"schema_version":1,"sequence":1,"media_type":"application/x.test","value_schema":"test.v1","payload":[1],"extra":true}}"#;
        assert!(serde_json::from_slice::<WorkerProcessMessage>(unknown).is_err());
    }
}
