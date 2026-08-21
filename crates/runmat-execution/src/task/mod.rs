use serde::{Deserialize, Serialize};

use crate::handle::OutputContract;
use crate::identity::{ArtifactId, ExecutionScopeId, PoolId, TaskId};
use crate::resource::ResourceRequest;
use crate::value::ValuePayload;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Callable {
    pub owner_identity: String,
    pub qualified_name: String,
    pub entrypoint_digest: crate::Digest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryPolicy {
    Never,
    IdempotentInfrastructure,
    ExplicitlyIdempotent { max_attempts: u16 },
    TestPolicy { max_attempts: u16 },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TaskRequest {
    pub id: TaskId,
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub program_artifact_id: ArtifactId,
    pub callable: Callable,
    pub inputs: Vec<ValuePayload>,
    pub outputs: OutputContract,
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
}
