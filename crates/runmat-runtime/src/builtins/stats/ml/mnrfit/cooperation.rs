//! mnrfit's error boundary over the shared execution-owned CPU budget.

use super::{build_runtime_error, BuiltinResult, NAME};
use crate::context::RuntimeContext;
use crate::execution::CpuCooperationBudget;
use runmat_execution::cancellation::CancellationToken;

#[derive(Clone)]
pub(super) struct FitCancellation {
    token: Option<CancellationToken>,
    namespace: Option<String>,
}

impl FitCancellation {
    pub(super) fn new(runtime: Option<&RuntimeContext>) -> Self {
        Self {
            token: runtime.map(RuntimeContext::cancellation),
            namespace: runtime.map(RuntimeContext::error_namespace),
        }
    }

    pub(super) fn check(&self) -> BuiltinResult<()> {
        if self
            .token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
        {
            let namespace = self.namespace.as_deref().unwrap_or("RunMat");
            return Err(build_runtime_error("mnrfit: execution was cancelled")
                .with_identifier(format!("{namespace}:Cancelled"))
                .with_builtin(NAME)
                .build());
        }
        Ok(())
    }
}

pub(super) struct FitCooperation {
    cancellation: FitCancellation,
    budget: Option<CpuCooperationBudget>,
}

impl FitCooperation {
    pub(super) fn new(runtime: Option<&RuntimeContext>) -> Self {
        let cancellation = FitCancellation::new(runtime);
        let budget = runtime.map(|runtime| {
            CpuCooperationBudget::new(
                runtime.execution().clone(),
                cancellation
                    .token
                    .as_ref()
                    .expect("runtime owns invocation token")
                    .clone(),
            )
        });
        Self {
            cancellation,
            budget,
        }
    }

    pub(super) fn cancellation(&self) -> FitCancellation {
        self.cancellation.clone()
    }

    pub(super) fn check(&self) -> BuiltinResult<()> {
        self.cancellation.check()
    }

    #[cfg(test)]
    pub(super) fn with_budget(runtime: &RuntimeContext, budget: CpuCooperationBudget) -> Self {
        Self {
            cancellation: FitCancellation::new(Some(runtime)),
            budget: Some(budget),
        }
    }

    pub(super) async fn completed(&mut self, units: u64) -> BuiltinResult<()> {
        self.check()?;
        if let Some(budget) = self.budget.as_mut() {
            let result = budget.checkpoint(units).await;
            self.check()?;
            result.map_err(crate::execution::ExecutionServiceError::into_runtime_error)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::rc::Rc;

    #[test]
    fn active_fit_keeps_its_invocation_token_across_context_rebinding() {
        let original = CancellationSource::new();
        let context = RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            original.token(),
        );
        let fit = FitCooperation::new(Some(&context));
        let replacement = CancellationSource::new();
        context.replace_cancellation_token(replacement.token());
        replacement.cancel(CancellationReason::User);
        fit.check()
            .expect("later generation must not cancel the active fit");
        original.cancel(CancellationReason::User);
        assert!(fit.check().is_err());
    }
}
