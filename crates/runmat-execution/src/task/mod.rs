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
#[serde(deny_unknown_fields)]
pub struct TaskRequest {
    pub id: TaskId,
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub program_artifact_id: ArtifactId,
    pub callable: Callable,
    pub inputs: Vec<ValuePayload>,
    pub environment: crate::EnvironmentSnapshot,
    #[serde(deserialize_with = "crate::console::deserialize_controls")]
    pub console: Option<crate::console::ConsoleLimits>,
    pub outputs: OutputContract,
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
}

impl TaskRequest {
    /// Admit the original context component. Whole execution storage remains separate.
    pub fn validate_invocation_reservation(&self) -> Result<(), crate::ContractError> {
        let minimum = crate::console::invocation_memory_bytes(&self.environment, self.console)?;
        let console = self
            .console
            .map(|limits| limits.storage_profile())
            .transpose()?;
        if minimum > self.resources.memory_bytes {
            return Err(crate::ContractError::invalid(
                "invocation context",
                "exceeds task memory reservation",
            ));
        }
        if console
            .is_some_and(|profile| profile.max_encoded_bytes > self.resources.max_artifact_bytes)
        {
            return Err(crate::ContractError::invalid(
                "invocation console",
                "exceeds task artifact reservation",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod console_tests;
