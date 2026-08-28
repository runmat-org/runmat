use std::fmt;
use std::sync::{Arc, Mutex};

use runmat_execution::identity::WorkerId;
use runmat_execution::{ExecutionScopeId, PoolId, ProgramRevision, TaskId};
use runmat_execution_runner::AttemptRequest;
use runmat_execution_runner::Driver;
use runmat_test_runner::worker::WorkerSessionId;

pub struct ExecutionWorkerSession<S> {
    pub(crate) id: WorkerSessionId,
    pub(crate) inner: S,
    pub(crate) driver: Arc<Mutex<Driver>>,
    pub(crate) scope_id: ExecutionScopeId,
    pub(crate) pool_id: PoolId,
    pub(crate) worker_id: WorkerId,
    pub(crate) revision: ProgramRevision,
    pub(crate) active: Arc<Mutex<Option<AttemptRequest>>>,
}

impl<S: Clone> Clone for ExecutionWorkerSession<S> {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            inner: self.inner.clone(),
            driver: Arc::clone(&self.driver),
            scope_id: self.scope_id,
            pool_id: self.pool_id,
            worker_id: self.worker_id,
            revision: self.revision.clone(),
            active: Arc::clone(&self.active),
        }
    }
}

impl<S> ExecutionWorkerSession<S> {
    pub fn id(&self) -> &WorkerSessionId {
        &self.id
    }

    pub fn program_revision(&self) -> &ProgramRevision {
        &self.revision
    }

    pub(crate) fn task_id(&self, test_id: &str, attempt: u32) -> TaskId {
        TaskId::derive(&[
            self.scope_id.bytes(),
            test_id.as_bytes(),
            &attempt.to_be_bytes(),
        ])
    }
}

impl<S> fmt::Debug for ExecutionWorkerSession<S> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExecutionWorkerSession")
            .field("id", &self.id)
            .field("scope_id", &self.scope_id)
            .field("pool_id", &self.pool_id)
            .field("worker_id", &self.worker_id)
            .field("revision", &self.revision)
            .finish_non_exhaustive()
    }
}

impl<S> PartialEq for ExecutionWorkerSession<S> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<S> Eq for ExecutionWorkerSession<S> {}
