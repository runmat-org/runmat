use std::future::Future;
use std::pin::Pin;

use super::{
    BackendCapabilities, BackendError, CancelRequest, ExecutionRequest, SpawnRequest,
    WorkerExecution,
};

pub type BackendFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, BackendError>> + 'a>>;

pub trait WorkerBackend {
    type Session: Clone + std::fmt::Debug + Eq;

    fn capabilities(&self) -> BackendCapabilities;
    fn spawn<'a>(&'a self, request: SpawnRequest) -> BackendFuture<'a, Self::Session>;
    fn execute<'a>(
        &'a self,
        session: &'a Self::Session,
        request: ExecutionRequest,
    ) -> BackendFuture<'a, WorkerExecution>;
    /// Carry the original scheduler allocation to a physical worker adapter.
    /// Backends without an isolated process retain their existing execution path.
    fn execute_admitted<'a>(
        &'a self,
        session: &'a Self::Session,
        request: ExecutionRequest,
        _assignment: runmat_execution::ProgramExecutionAssignment,
        _resources: runmat_execution::resource::ResourceRequest,
    ) -> BackendFuture<'a, WorkerExecution> {
        self.execute(session, request)
    }
    fn cancel<'a>(
        &'a self,
        session: &'a Self::Session,
        request: CancelRequest,
    ) -> BackendFuture<'a, Option<WorkerExecution>>;
    fn terminate<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()>;
    fn shutdown<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()>;
    /// Transfer bounded raw application output after this worker has retired.
    /// Backends without a native process have no process stream to report.
    fn take_process_output<'a>(
        &'a self,
        _session: &'a Self::Session,
    ) -> BackendFuture<'a, Option<runmat_test::event::WorkerProcessOutput>> {
        Box::pin(async { Ok(None) })
    }
}
