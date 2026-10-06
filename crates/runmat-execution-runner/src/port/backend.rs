use std::future::Future;
use std::pin::Pin;

use runmat_execution::identity::{AttemptId, WorkerId};
use runmat_execution::TaskId;

use crate::task::{AttemptReport, AttemptRequest};
use crate::RunnerResult;

pub type PortFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackendReport {
    pub attempt_id: AttemptId,
    pub task_id: TaskId,
    pub worker_id: WorkerId,
    pub driver_fence: u64,
    pub report: AttemptReport,
    /// Original provider evidence; absent when the provider cannot establish it.
    #[serde(deserialize_with = "required_option")]
    pub execution: Option<Box<crate::task::AttemptExecutionObservation>>,
    /// Original accounting for an unsuccessful terminal report. Absence leaves
    /// consumption unknown; successful accounting belongs to AttemptSuccess.
    #[serde(deserialize_with = "required_option")]
    pub unsuccessful_resources: Option<Box<crate::task::AttemptResourceAccounting>>,
}

impl BackendReport {
    pub fn for_request(request: &AttemptRequest, report: AttemptReport) -> Self {
        Self {
            attempt_id: request.id,
            task_id: request.task_id,
            worker_id: request.worker_id,
            driver_fence: request.driver_fence,
            report,
            execution: None,
            unsuccessful_resources: None,
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

fn required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    serde::Deserialize::deserialize(deserializer)
}
