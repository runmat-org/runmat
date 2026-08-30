use serde::{Deserialize, Serialize};

mod assignment;
mod callable;
mod failure;
mod parallel;

pub use assignment::ProgramExecutionAssignment;
pub use callable::ProgramCallable;
pub use failure::{ProgramCallFrame, ProgramRuntimeFailure, ProgramSourceSpan};
pub use parallel::{
    ParallelChunk, ParallelRandomStream, ParallelRandomnessContext, ParallelTaskContext,
    ParallelTaskGraph, ProgramInvocationContext,
};

use crate::handle::OutputContract;
use crate::identity::{ArtifactId, ExecutionScopeId, PoolId, TaskId};
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
pub struct TaskRequest {
    pub id: TaskId,
    pub scope_id: ExecutionScopeId,
    pub pool_id: PoolId,
    pub program_artifact_id: ArtifactId,
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
    pub outputs: OutputContract,
    pub resources: ResourceRequest,
    pub retry: RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
}
