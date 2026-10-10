//! Tabular invocations use the installed execution service for suspension and
//! CPU work accounting; this module owns no scheduling policy.

use std::future::Future;

use crate::context::RuntimeContext;
use crate::execution::staged_file::{self, StagedCpuCooperation};
#[cfg(test)]
use crate::execution::CpuCooperationBudget;
use crate::BuiltinResult;

pub(super) struct TabularCooperation<'a> {
    cooperation: StagedCpuCooperation<'a>,
}

impl staged_file::StagedWriteCooperation for TabularCooperation<'_> {
    async fn completed(&mut self, work: usize) -> BuiltinResult<()> {
        TabularCooperation::completed(self, work).await
    }
}

impl<'a> TabularCooperation<'a> {
    pub(super) fn new(runtime: Option<&'a RuntimeContext>, builtin: &'static str) -> Self {
        Self {
            cooperation: StagedCpuCooperation::new(runtime, builtin),
        }
    }

    #[cfg(test)]
    pub(super) fn with_test_budget(
        runtime: &'a RuntimeContext,
        builtin: &'static str,
        budget: CpuCooperationBudget,
    ) -> Self {
        Self {
            cooperation: StagedCpuCooperation::with_test_budget(runtime, builtin, budget),
        }
    }

    pub(super) async fn completed(&mut self, work: usize) -> BuiltinResult<()> {
        self.cooperation.completed(work).await
    }
}

pub(super) async fn await_io<T, F>(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    work: F,
) -> BuiltinResult<T>
where
    F: Future<Output = T>,
{
    staged_file::await_cancellable(runtime, builtin, work).await
}
