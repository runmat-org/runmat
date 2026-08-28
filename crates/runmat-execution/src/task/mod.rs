use serde::{Deserialize, Serialize};

mod callable;

pub use callable::ProgramCallable;

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

impl Callable {
    /// Builds scheduler metadata from the exact callable admitted by the
    /// program worker. The scheduler may display and hash this metadata, but
    /// the typed [`ProgramCallable`] remains execution authority.
    pub fn for_program(owner_identity: impl Into<String>, callable: &ProgramCallable) -> Self {
        Self {
            owner_identity: owner_identity.into(),
            qualified_name: callable.display_name(),
            entrypoint_digest: callable.identity_digest(),
        }
    }

    pub fn identifies_program(&self, callable: &ProgramCallable) -> bool {
        self.qualified_name == callable.display_name()
            && self.entrypoint_digest == callable.identity_digest()
    }
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
