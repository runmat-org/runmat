//! Await ownership retains the issuing Value metadata, never a reconstructed
//! wire contract. Services remain responsible for scope/generation/state joins.

use runmat_execution::{ExecutionScopeId, FutureHandle, FutureId, OutputContract};
use runmat_value::{SharedFutureHandle, SharedTaskHandle, Value};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FutureCompletionRef<'a> {
    pub id: FutureId,
    pub scope_id: ExecutionScopeId,
    pub outputs: &'a OutputContract,
}

impl<'a> From<&'a FutureHandle> for FutureCompletionRef<'a> {
    fn from(handle: &'a FutureHandle) -> Self {
        Self {
            id: handle.id,
            scope_id: handle.scope_id,
            outputs: &handle.outputs,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum FutureSource {
    Future(SharedFutureHandle),
    Task(SharedTaskHandle),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FutureExecutionIdentity {
    pub id: FutureId,
    pub scope_id: ExecutionScopeId,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FutureExecutionHandle {
    identity: FutureExecutionIdentity,
    source: FutureSource,
}

impl FutureExecutionHandle {
    /// Called only after the service joins the task's live generation to its
    /// stored future. This constructor alone confers no execution authority.
    pub fn from_task(task: SharedTaskHandle, future_id: FutureId) -> Self {
        Self {
            identity: FutureExecutionIdentity {
                id: future_id,
                scope_id: task.scope_id,
            },
            source: FutureSource::Task(task),
        }
    }
    pub fn completion_ref(&self) -> FutureCompletionRef<'_> {
        let outputs = match &self.source {
            FutureSource::Future(handle) => &handle.outputs,
            FutureSource::Task(handle) => &handle.outputs,
        };
        FutureCompletionRef {
            id: self.id,
            scope_id: self.scope_id,
            outputs,
        }
    }
    pub fn retained_value(&self) -> Value {
        match &self.source {
            FutureSource::Future(handle) => Value::Future(handle.clone()),
            FutureSource::Task(handle) => Value::Task(handle.clone()),
        }
    }
}
impl From<SharedFutureHandle> for FutureExecutionHandle {
    fn from(handle: SharedFutureHandle) -> Self {
        Self {
            identity: FutureExecutionIdentity {
                id: handle.id,
                scope_id: handle.scope_id,
            },
            source: FutureSource::Future(handle),
        }
    }
}
impl From<FutureHandle> for FutureExecutionHandle {
    fn from(handle: FutureHandle) -> Self {
        SharedFutureHandle::new(handle).into()
    }
}
impl<'a> From<&'a FutureExecutionHandle> for FutureCompletionRef<'a> {
    fn from(handle: &'a FutureExecutionHandle) -> Self {
        handle.completion_ref()
    }
}
impl std::ops::Deref for FutureExecutionHandle {
    type Target = FutureExecutionIdentity;
    fn deref(&self) -> &Self::Target {
        &self.identity
    }
}
