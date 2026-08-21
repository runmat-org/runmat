use std::collections::VecDeque;

use crate::port::{BackendPort, BackendReport, PortFuture};
use crate::task::{AttemptReport, AttemptRequest};
use crate::{RunnerError, RunnerResult};

#[derive(Clone, Debug, Default)]
pub struct ScriptedBackend {
    reports: VecDeque<AttemptReport>,
    pub launched: Vec<AttemptRequest>,
    pub cancelled: Vec<AttemptRequest>,
}

impl ScriptedBackend {
    pub fn new(reports: impl IntoIterator<Item = AttemptReport>) -> Self {
        Self {
            reports: reports.into_iter().collect(),
            launched: Vec::new(),
            cancelled: Vec::new(),
        }
    }
}

impl BackendPort for ScriptedBackend {
    fn launch<'a>(
        &'a mut self,
        request: AttemptRequest,
    ) -> PortFuture<'a, RunnerResult<BackendReport>> {
        Box::pin(async move {
            self.launched.push(request.clone());
            let report = self
                .reports
                .pop_front()
                .ok_or_else(|| RunnerError::Backend("scripted backend is exhausted".into()))?;
            Ok(BackendReport::for_request(&request, report))
        })
    }

    fn cancel<'a>(&'a mut self, request: &'a AttemptRequest) -> PortFuture<'a, RunnerResult<()>> {
        Box::pin(async move {
            self.cancelled.push(request.clone());
            Ok(())
        })
    }
}
