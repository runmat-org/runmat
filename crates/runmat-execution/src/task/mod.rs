use serde::{Deserialize, Serialize};

mod assignment;
mod callable;
mod failure;
mod parallel;
mod owned_builtin;

pub use assignment::ProgramExecutionAssignment;
pub use callable::ProgramCallable;
pub use failure::{
    ProgramCallFrame, ProgramDetachedDiagnosticSource, ProgramRuntimeFailure, ProgramSourceSpan,
};
pub use owned_builtin::{
    OwnedBuiltinBinding, OwnedBuiltinDenseOutputContract, OwnedBuiltinResponseContract,
    OwnedBuiltinWorkRequest, OwnedBuiltinWorkResponse, OWNED_BUILTIN_FAILURE_WIRE_BYTES,
    OWNED_BUILTIN_MAX_RESPONSE_WIRE_BYTES, OWNED_BUILTIN_VALUE_LIMITS,
    OWNED_BUILTIN_WORK_SCHEMA_VERSION,
};
pub use parallel::{
    deterministic_iteration_seed, ParallelChunk, ParallelRandomStream, ParallelRandomnessContext,
    ParallelTaskContext, ParallelTaskGraph, ProgramInvocationContext, RUNMAT_MT_DEFAULT_SEED,
};

pub const TASK_REQUEST_SCHEMA_VERSION: u16 = crate::schema::EXECUTION_SCHEMA_V2;

use crate::handle::OutputContract;
use crate::identity::{ArtifactId, Digest, ExecutionScopeId, PoolId, TaskId};
use crate::resource::ResourceRequest;
use crate::value::ValuePayload;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Callable {
    pub owner_identity: String,
    pub program: ProgramCallable,
}

impl Callable {
    /// Builds scheduler metadata from the exact callable admitted by the
    /// program worker. The scheduler may display and hash this metadata, but
    /// the typed [`ProgramCallable`] remains execution authority.
    pub fn for_program(owner_identity: impl Into<String>, callable: &ProgramCallable) -> Self {
        Self {
            owner_identity: owner_identity.into(),
            program: callable.clone(),
        }
    }

    pub fn identifies_program(&self, callable: &ProgramCallable) -> bool {
        self.program == *callable
    }

    pub fn qualified_name(&self) -> String {
        self.program.display_name()
    }

    pub fn entrypoint_digest(&self) -> crate::Digest {
        self.program.identity_digest()
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
#[serde(
    tag = "kind",
    content = "identity",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum TaskExecutableIdentity {
    ProgramArtifact(ArtifactId),
    /// Exact digest of an admitted canonical binding and catalog revision.
    OwnedBuiltin(Digest),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskRequest {
    pub schema_version: u16,
    pub id: TaskId,
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub executable: TaskExecutableIdentity,
    pub callable: Callable,
    /// Compiler-derived runtime, target, interop, stack, and trust contract.
    /// Scheduling must satisfy it before an attempt receives resources.
    pub host: crate::host::ExecutionHostRequirement,
    /// Typed per-task invocation metadata. This is separate from the immutable
    /// program artifact so retries and parallel chunks can share one compiled
    /// product without aliasing their execution context.
    #[serde(default)]
    pub invocation_context: ProgramInvocationContext,
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

    pub fn validate_executable(&self) -> Result<(), crate::ContractError> {
        if self.schema_version != TASK_REQUEST_SCHEMA_VERSION {
            return Err(crate::ContractError::invalid(
                "task request",
                "unsupported executable-work schema",
            ));
        }
        if matches!(&self.executable, TaskExecutableIdentity::OwnedBuiltin(_))
            && (!matches!(&self.callable.program, ProgramCallable::Builtin { .. })
                || self.outputs.owned_work.is_none())
        {
            return Err(crate::ContractError::invalid(
                "task request",
                "owned builtin identity lacks its exact callable or output contract",
            ));
        }
        if !matches!(&self.executable, TaskExecutableIdentity::OwnedBuiltin(_))
            && self.outputs.owned_work.is_some()
        {
            return Err(crate::ContractError::invalid(
                "task request",
                "program artifact cannot carry an owned builtin output contract",
            ));
        }
        if let Some(contract) = &self.outputs.owned_work {
            contract.validate(self.outputs.requested_outputs)?;
        }
        Ok(())
    }

    pub fn program_artifact_id(&self) -> Option<ArtifactId> {
        match &self.executable {
            TaskExecutableIdentity::ProgramArtifact(id) => Some(*id),
            TaskExecutableIdentity::OwnedBuiltin(_) => None,
        }
    }
}

#[cfg(test)]
mod console_tests;
