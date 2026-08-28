use crate::port::{BackendPort, BackendReport, PortFuture};
use crate::task::{AttemptReport, AttemptRequest};
use crate::RunnerResult;

pub struct SerialBackend<F> {
    execute: F,
}

impl<F> SerialBackend<F> {
    pub fn new(execute: F) -> Self {
        Self { execute }
    }
}

impl<F> BackendPort for SerialBackend<F>
where
    F: FnMut(&AttemptRequest) -> RunnerResult<AttemptReport>,
{
    fn launch<'a>(
        &'a mut self,
        request: AttemptRequest,
    ) -> PortFuture<'a, RunnerResult<BackendReport>> {
        Box::pin(async move {
            let report = (self.execute)(&request)?;
            Ok(BackendReport::for_request(&request, report))
        })
    }

    fn cancel<'a>(&'a mut self, _request: &'a AttemptRequest) -> PortFuture<'a, RunnerResult<()>> {
        Box::pin(async { Ok(()) })
    }
}
