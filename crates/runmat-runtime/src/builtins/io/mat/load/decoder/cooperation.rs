//! MAT decode work is charged to the invocation execution service. The
//! captured filesystem owner remains authoritative between codec chunks.

use super::super::super::authority::MatFsPath;
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult};
use std::future::Future;

pub(super) struct MatDecodeCooperation<'a> {
    budget: Option<CpuCooperationBudget>,
    source: Option<&'a MatFsPath>,
}

impl<'a> MatDecodeCooperation<'a> {
    pub(super) fn new(runtime: Option<&RuntimeContext>, source: Option<&'a MatFsPath>) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            source,
        }
    }

    #[cfg(test)]
    pub(super) fn with_budget(budget: CpuCooperationBudget) -> Self {
        Self {
            budget: Some(budget),
            source: None,
        }
    }

    pub(super) async fn completed(&mut self, bytes: usize) -> BuiltinResult<()> {
        self.check_source()?;
        if let Some(budget) = self.budget.as_mut() {
            budget
                .checkpoint(bytes as u64)
                .await
                .map_err(runtime_error)?;
        }
        self.check_source()
    }

    pub(super) async fn await_io<T, E, F>(&mut self, future: F) -> BuiltinResult<Result<T, E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        self.completed(0).await?;
        let result = if let Some(budget) = self.budget.as_ref() {
            budget
                .await_cancellable(future)
                .await
                .map_err(runtime_error)?
        } else {
            future.await
        };
        if result.is_ok() {
            self.completed(0).await?;
        }
        Ok(result)
    }

    fn check_source(&self) -> BuiltinResult<()> {
        if let Some(source) = self.source {
            source.check_current().map_err(|error| {
                super::super::load_error_with_source(
                    &runmat_builtins::LOAD_ERROR_IO,
                    "load: captured MAT source was revoked during decoding",
                    error,
                )
            })?;
        }
        Ok(())
    }
}

fn runtime_error(error: ExecutionServiceError) -> crate::RuntimeError {
    match error {
        cancelled @ ExecutionServiceError::Cancelled => {
            build_runtime_error("load: MAT decoding was cancelled")
                .with_identifier("RunMat:Cancelled")
                .with_builtin("load")
                .with_source(cancelled)
                .build()
        }
        other => other.into_runtime_error(),
    }
}
