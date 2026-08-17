use serde::{Deserialize, Serialize};

pub use runmat_execution_artifact::{
    ProgramExecutionRequest as WorkerRequest, ProgramExecutionResponse as WorkerResponse,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};

pub const NATIVE_WORKER_MESSAGE_SCHEMA_V1: u16 = 1;
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
        if self.schema_version != NATIVE_WORKER_MESSAGE_SCHEMA_V1
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
    Progress { progress: ProgramProgress },
    Completed { response: WorkerResponse },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StoredProgram {
    pub recipe: runmat_execution_artifact::ProgramBuildRecipe,
    pub artifact: runmat_execution_artifact::ProgramArtifact,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_envelope_is_bounded_and_strictly_shaped() {
        let progress = ProgramProgress {
            schema_version: NATIVE_WORKER_MESSAGE_SCHEMA_V1,
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
