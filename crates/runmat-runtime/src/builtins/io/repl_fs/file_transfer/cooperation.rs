//! Invocation-owned cooperation and cancellation across file-transfer awaits.

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult};
use std::future::Future;

use super::authority::TransferOperation;

pub(super) struct TransferCooperation {
    operation: TransferOperation,
    budget: Option<CpuCooperationBudget>,
}

impl TransferCooperation {
    pub(super) fn new(operation: TransferOperation, runtime: Option<&RuntimeContext>) -> Self {
        Self {
            operation,
            budget: runtime.map(RuntimeContext::cpu_cooperation),
        }
    }

    #[cfg(test)]
    pub(super) fn with_test_budget(
        operation: TransferOperation,
        budget: CpuCooperationBudget,
    ) -> Self {
        Self {
            operation,
            budget: Some(budget),
        }
    }

    pub(super) async fn checkpoint(&mut self) -> BuiltinResult<()> {
        let completed_work = match self.operation {
            TransferOperation::Copy => 0,
            TransferOperation::Move => 1,
        };
        self.completed_bytes(completed_work).await
    }

    async fn completed_bytes(&mut self, bytes: u64) -> BuiltinResult<()> {
        self.raw_checkpoint(bytes)
            .await
            .map_err(|error| runtime_error(self.operation, error))?;
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
                .map_err(|error| runtime_error(self.operation, error))?
        } else {
            future.await
        };
        if result.is_ok() {
            self.checkpoint().await?;
        }
        result
    }
}

fn runtime_error(
    operation: TransferOperation,
    error: ExecutionServiceError,
) -> crate::RuntimeError {
    match error {
        cancelled @ ExecutionServiceError::Cancelled => {
            build_runtime_error(format!("{}: operation was cancelled", operation.name()))
                .with_identifier("RunMat:Cancelled")
                .with_builtin(operation.name())
                .with_source(cancelled)
                .build()
        }
        other => other.into_runtime_error(),
    }
}
