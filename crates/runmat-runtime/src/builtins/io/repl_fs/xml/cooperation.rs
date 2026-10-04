//! XML work units use the invocation's execution service for yield/cancel policy.

use std::future::Future;

use runmat_builtins::{XmlDiagnosticKind, XmlOperation};

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

pub(super) struct XmlCooperation {
    budget: Option<CpuCooperationBudget>,
    operation: XmlOperation,
}

impl XmlCooperation {
    pub(super) fn operation(&self) -> XmlOperation {
        self.operation
    }

    pub(super) fn is_execution_error(&self, error: &RuntimeError) -> bool {
        let mut source = std::error::Error::source(error);
        while let Some(cause) = source {
            if cause.downcast_ref::<ExecutionServiceError>().is_some() {
                return true;
            }
            source = cause.source();
        }
        false
    }
    pub(super) fn new(runtime: Option<&RuntimeContext>, operation: XmlOperation) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            operation,
        }
    }

    #[cfg(test)]
    pub(super) fn with_budget(budget: CpuCooperationBudget, operation: XmlOperation) -> Self {
        Self {
            budget: Some(budget),
            operation,
        }
    }

    pub(super) async fn completed(&mut self, units: usize) -> BuiltinResult<()> {
        if let Some(budget) = self.budget.as_mut() {
            budget
                .checkpoint(units as u64)
                .await
                .map_err(|err| self.map_error(err))?;
        }
        Ok(())
    }

    pub(super) async fn await_io<T, E, F>(&mut self, future: F) -> BuiltinResult<Result<T, E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        // Check the invocation before polling an I/O future, including an
        // implementation that would otherwise complete in its first poll.
        self.completed(0).await?;
        match self.budget.as_ref() {
            Some(budget) => budget
                .await_cancellable(future)
                .await
                .map_err(|err| self.map_error(err)),
            None => Ok(future.await),
        }
    }

    fn map_error(&self, error: ExecutionServiceError) -> RuntimeError {
        match error {
            cancelled @ ExecutionServiceError::Cancelled => build_runtime_error(format!(
                "{}: execution was cancelled",
                self.operation.name()
            ))
            .with_identifier(self.operation.identifier(XmlDiagnosticKind::Cancelled))
            .with_builtin(self.operation.name())
            .with_source(cancelled)
            .build(),
            other => {
                let message = format!(
                    "{}: execution service failed: {other}",
                    self.operation.name()
                );
                build_runtime_error(message)
                    .with_identifier(self.operation.identifier(XmlDiagnosticKind::Execution))
                    .with_builtin(self.operation.name())
                    .with_source(other)
                    .build()
            }
        }
    }
}
