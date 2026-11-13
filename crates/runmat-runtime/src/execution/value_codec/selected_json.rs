//! Original-budget continuation around canonical selected-capture JSON. This
//! owner is data-only until ordinary archive/VM admission authenticates code.
use crate::{context::RuntimeContext, execution::CpuCooperationBudget, BuiltinResult};
mod code;
mod json;
mod read_graph;
#[cfg(test)]
mod tests;
pub(crate) use json::{Reader, Writer};

pub(crate) struct Cooperation<'a> {
    runtime: &'a RuntimeContext,
    budget: &'a mut CpuCooperationBudget,
}
impl<'a> Cooperation<'a> {
    pub(crate) fn new(
        runtime: &'a RuntimeContext,
        budget: &'a mut CpuCooperationBudget,
    ) -> BuiltinResult<Self> {
        if !budget
            .cancellation_token()
            .same_generation(&runtime.cancellation())
        {
            return Err(error("selected JSON has a foreign cooperation generation"));
        }
        crate::context::host_allocation::require_unbounded_workspace(
            Some(runtime),
            "selected capture JSON",
            "selected capture JSON has no complete finite byte/metadata workspace bound",
        )?;
        Ok(Self { runtime, budget })
    }
    pub(crate) async fn completed(&mut self, units: u64) -> BuiltinResult<()> {
        self.budget
            .checkpoint(units)
            .await
            .map_err(|source| source.into_runtime_error_in_context(Some(self.runtime)))
    }
}

impl Writer {
    pub(crate) async fn graph(
        &mut self,
        graph: &runmat_execution::value::PayloadGraph,
        cooperation: &mut Cooperation<'_>,
    ) -> BuiltinResult<()> {
        use runmat_execution::value::{SelectedGraphJsonCursor, SelectedGraphJsonProgress};
        let mut cursor = SelectedGraphJsonCursor::new(graph)
            .map_err(|source| with_source("cannot prepare selected graph JSON", source))?;
        loop {
            let progress = cursor
                .advance(
                    &mut self.bytes,
                    std::num::NonZeroUsize::new(256).expect("fixed nonzero codec work slice"),
                )
                .map_err(|source| with_source("cannot encode selected graph JSON", source))?;
            let (completed, done) = match progress {
                SelectedGraphJsonProgress::Pending { completed_work } => (completed_work, false),
                SelectedGraphJsonProgress::Complete { completed_work } => (completed_work, true),
            };
            cooperation.completed(completed as u64).await?;
            if done {
                return Ok(());
            }
        }
    }
}
fn error(message: impl std::fmt::Display) -> crate::RuntimeError {
    crate::build_runtime_error(format!("selected capture JSON: {message}"))
        .with_identifier("RunMat:Execution:InvalidSelectedGraphJson")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
fn with_source(
    message: impl std::fmt::Display,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> crate::RuntimeError {
    crate::build_runtime_error(format!("selected capture JSON: {message}"))
        .with_identifier("RunMat:Execution:InvalidSelectedGraphJson")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .with_source(source)
        .build()
}
