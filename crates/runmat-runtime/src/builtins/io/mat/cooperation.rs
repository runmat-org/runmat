//! MAT codec work uses the invocation execution service's yield/cancellation
//! policy and the exact captured filesystem owner across awaits.

use super::authority::MatFsPath;
use crate::context::host_effect::HostEffectResourceLivenessLease;
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_builtins::BuiltinErrorDescriptor;
use std::future::Future;
use std::rc::Rc;
use std::sync::Arc;

pub(in crate::builtins::io::mat) struct MatCodecCooperation<'a> {
    runtime: Option<&'a RuntimeContext>,
    budget: Option<Budget<'a>>,
    source: Option<&'a MatFsPath>,
    resource_liveness: Option<Arc<dyn HostEffectResourceLivenessLease>>,
    workspace_liveness: Option<Rc<dyn crate::context::RuntimeWorkspaceService>>,
    builtin: &'static str,
    io_error: &'static BuiltinErrorDescriptor,
}

enum Budget<'a> {
    Owned(CpuCooperationBudget),
    Borrowed(&'a mut CpuCooperationBudget),
}
impl std::ops::Deref for Budget<'_> {
    type Target = CpuCooperationBudget;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Owned(budget) => budget,
            Self::Borrowed(budget) => budget,
        }
    }
}
impl std::ops::DerefMut for Budget<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Self::Owned(budget) => budget,
            Self::Borrowed(budget) => budget,
        }
    }
}

impl<'a> MatCodecCooperation<'a> {
    /// Reuse the actual caller's pending work and original cancellation turn.
    pub(in crate::builtins::io::mat) fn from_original_budget(
        runtime: Option<&'a RuntimeContext>,
        budget: Option<&'a mut CpuCooperationBudget>,
        source: Option<&'a MatFsPath>,
        builtin: &'static str,
        io_error: &'static BuiltinErrorDescriptor,
    ) -> BuiltinResult<Self> {
        match (runtime, budget.as_deref()) {
            (Some(runtime), Some(budget))
                if budget
                    .cancellation_token()
                    .same_generation(&runtime.cancellation()) => {}
            (None, None) => {}
            _ => {
                return Err(build_runtime_error(
                    "MAT codec cooperation differs from the original invocation",
                )
                .with_identifier("RunMat:Execution:CooperationOwner")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build())
            }
        }
        Ok(Self {
            runtime,
            budget: budget.map(Budget::Borrowed),
            source,
            resource_liveness: None,
            workspace_liveness: None,
            builtin,
            io_error,
        })
    }

    pub(in crate::builtins::io::mat) fn new(
        runtime: Option<&'a RuntimeContext>,
        source: Option<&'a MatFsPath>,
        builtin: &'static str,
        io_error: &'static BuiltinErrorDescriptor,
    ) -> Self {
        Self {
            runtime,
            budget: runtime.map(|runtime| Budget::Owned(runtime.cpu_cooperation())),
            source,
            resource_liveness: None,
            workspace_liveness: None,
            builtin,
            io_error,
        }
    }

    #[cfg(test)]
    pub(in crate::builtins::io::mat) fn with_budget(
        budget: CpuCooperationBudget,
        builtin: &'static str,
        io_error: &'static BuiltinErrorDescriptor,
    ) -> Self {
        Self {
            runtime: None,
            budget: Some(Budget::Owned(budget)),
            source: None,
            resource_liveness: None,
            workspace_liveness: None,
            builtin,
            io_error,
        }
    }

    pub(in crate::builtins::io::mat) fn runtime(&self) -> Option<&'a RuntimeContext> {
        self.runtime
    }

    pub(in crate::builtins::io::mat) fn with_resource_liveness(
        mut self,
        resource_liveness: Option<Arc<dyn HostEffectResourceLivenessLease>>,
    ) -> Self {
        self.resource_liveness = resource_liveness;
        self
    }

    pub(in crate::builtins::io::mat) fn resource_liveness_lease(
        &self,
    ) -> Option<Arc<dyn HostEffectResourceLivenessLease>> {
        self.resource_liveness.clone()
    }

    pub(in crate::builtins::io::mat) fn with_workspace_liveness(
        mut self,
        workspace_liveness: Option<Rc<dyn crate::context::RuntimeWorkspaceService>>,
    ) -> Self {
        self.workspace_liveness = workspace_liveness;
        self
    }

    pub(in crate::builtins::io::mat) async fn completed(
        &mut self,
        bytes: usize,
    ) -> BuiltinResult<()> {
        self.check_source()?;
        if let Some(budget) = self.budget.as_mut() {
            let result = budget.checkpoint(bytes as u64).await;
            result.map_err(|error| self.runtime_error(error))?;
        }
        self.check_source()
    }

    pub(in crate::builtins::io::mat) async fn await_io<T, E, F>(
        &mut self,
        future: F,
    ) -> BuiltinResult<Result<T, E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        self.completed(0).await?;
        let result = if let Some(budget) = self.budget.as_ref() {
            budget
                .await_cancellable(future)
                .await
                .map_err(|error| self.runtime_error(error))?
        } else {
            future.await
        };
        if result.is_ok() {
            self.completed(0).await?;
        }
        Ok(result)
    }

    fn check_source(&self) -> BuiltinResult<()> {
        if let Some(service) = self.workspace_liveness.as_ref() {
            let liveness = match self.runtime {
                Some(runtime) => service.ensure_effect_authority_with_context(runtime),
                None => service.ensure_effect_authority(),
            };
            liveness.map_err(|mut error| {
                error.context = error.context.with_builtin(self.builtin);
                error
            })?;
        }
        if self
            .resource_liveness
            .as_ref()
            .is_some_and(|lease| !lease.is_current())
        {
            return Err(crate::context::host_effect::stale_resource());
        }
        if let Some(source) = self.source {
            source.check_current().map_err(|error| {
                let mut builder = build_runtime_error(format!(
                    "{}: captured MAT source was revoked during codec work",
                    self.builtin
                ))
                .with_builtin(self.builtin)
                .with_source(error);
                if let Some(identifier) = self.io_error.identifier {
                    builder = builder.with_identifier(identifier);
                }
                builder.build()
            })?;
        }
        Ok(())
    }

    fn runtime_error(&self, error: ExecutionServiceError) -> RuntimeError {
        match error {
            cancelled @ ExecutionServiceError::Cancelled => {
                build_runtime_error(format!("{}: MAT codec work was cancelled", self.builtin))
                    .with_identifier("RunMat:Cancelled")
                    .with_builtin(self.builtin)
                    .with_source(cancelled)
                    .build()
            }
            other => other.into_runtime_error(),
        }
    }
}
