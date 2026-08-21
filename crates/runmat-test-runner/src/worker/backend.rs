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
    fn cancel<'a>(
        &'a self,
        session: &'a Self::Session,
        request: CancelRequest,
    ) -> BackendFuture<'a, Option<WorkerExecution>>;
    fn terminate<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()>;
    fn shutdown<'a>(&'a self, session: &'a Self::Session) -> BackendFuture<'a, ()>;
}
