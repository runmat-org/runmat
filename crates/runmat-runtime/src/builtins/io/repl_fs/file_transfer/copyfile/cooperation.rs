//! Invocation-owned cancellation across copyfile's provider awaits.

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult};
use std::future::Future;

pub(super) struct CopyCooperation {
    budget: Option<CpuCooperationBudget>,
}

impl CopyCooperation {
    pub(super) fn new(runtime: Option<&RuntimeContext>) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
        }
    }

    pub(super) async fn checkpoint(&mut self) -> BuiltinResult<()> {
        self.completed_bytes(0).await
    }

    async fn completed_bytes(&mut self, bytes: u64) -> BuiltinResult<()> {
        self.raw_checkpoint(bytes).await.map_err(runtime_error)?;
        Ok(())
    }

    pub(super) async fn raw_checkpoint(&mut self, bytes: u64) -> Result<(), ExecutionServiceError> {
        if let Some(budget) = self.budget.as_mut() {
            budget.checkpoint(bytes).await?;
        }
        Ok(())
    }

    pub(super) async fn await_operation<T>(
        &mut self,
        future: impl Future<Output = BuiltinResult<T>>,
    ) -> BuiltinResult<T> {
        self.checkpoint().await?;
        let result = if let Some(budget) = self.budget.as_ref() {
            budget
                .await_cancellable(future)
                .await
                .map_err(runtime_error)?
        } else {
            future.await
        };
        if result.is_ok() {
            self.checkpoint().await?;
        }
        result
    }
}

fn runtime_error(error: ExecutionServiceError) -> crate::RuntimeError {
    match error {
        cancelled @ ExecutionServiceError::Cancelled => {
            build_runtime_error("copyfile: operation was cancelled")
                .with_identifier("RunMat:Cancelled")
                .with_builtin("copyfile")
                .with_source(cancelled)
                .build()
        }
        other => other.into_runtime_error(),
    }
}
