use std::future::Future;
use std::pin::Pin;

use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::TaskId;

use crate::task::{AttemptReport, AttemptRequest};
use crate::RunnerResult;

pub type PortFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendReport {
    pub attempt_id: AttemptId,
    pub task_id: TaskId,
    pub worker_id: WorkerId,
    pub driver_fence: u64,
    pub report: AttemptReport,
}

impl BackendReport {
    pub fn for_request(request: &AttemptRequest, report: AttemptReport) -> Self {
        Self {
            attempt_id: request.id,
            task_id: request.task_id,
            worker_id: request.worker_id,
            driver_fence: request.driver_fence,
            report,
        }
    }
}

pub trait BackendPort {
    fn launch<'a>(
        &'a mut self,
        request: AttemptRequest,
    ) -> PortFuture<'a, RunnerResult<BackendReport>>;

    fn cancel<'a>(&'a mut self, request: &'a AttemptRequest) -> PortFuture<'a, RunnerResult<()>>;
}
