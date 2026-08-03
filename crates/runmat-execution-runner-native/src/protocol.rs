use runmat_execution::value::ValuePayload;
use serde::{Deserialize, Serialize};

pub const PROTOCOL: &str = "runmat-local-execution-v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerRequest {
    pub protocol: String,
    pub program_digest: runmat_execution::Digest,
    pub program: Vec<u8>,
    pub function: usize,
    pub arguments: Vec<ValuePayload>,
    pub requested_outputs: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case", deny_unknown_fields)]
pub enum WorkerResponse {
    Success { value: ValuePayload },
    Failure { message: String },
}
