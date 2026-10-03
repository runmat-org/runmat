//! Audio-owned work accounting; the runtime execution service owns yield policy.

use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use std::future::Future;

pub(super) struct AudioCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
    builtin: &'static str,
    cancelled: bool,
}

impl AudioCooperation {
    pub(super) fn new(runtime: Option<&RuntimeContext>, builtin: &'static str) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            namespace: runtime.map(RuntimeContext::error_namespace),
            builtin,
            cancelled: false,
        }
    }

    #[cfg(test)]
    pub(super) fn with_budget(budget: CpuCooperationBudget, builtin: &'static str) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".into()),
            builtin,
            cancelled: false,
        }
    }

    /// True for a typed cancellation from either the invocation token or its
    /// execution service; no diagnostic-string inference is needed by feedback.
    pub(super) fn was_cancelled(&self) -> bool {
        self.cancelled
    }

    /// Suspend on provider I/O without losing the invocation's cancellation
    /// wakeup. The caller retains the original I/O error for its own descriptor.
    pub(super) async fn await_io<T, E, F>(&mut self, future: F) -> BuiltinResult<Result<T, E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        let Some(budget) = self.budget.as_ref() else {
            return Ok(future.await);
        };
        let outcome = budget.await_cancellable(future).await;
        outcome.map_err(|error| self.runtime_error(error))
    }

    pub(super) async fn completed(&mut self, units: usize) -> BuiltinResult<()> {
        let Some(budget) = self.budget.as_mut() else {
            return Ok(());
        };
        let outcome = budget.checkpoint(units as u64).await;
        outcome.map_err(|error| self.runtime_error(error))
    }

    fn runtime_error(&mut self, error: ExecutionServiceError) -> RuntimeError {
        match error {
            ExecutionServiceError::Cancelled => {
                self.cancelled = true;
                build_runtime_error(format!("{}: execution was cancelled", self.builtin))
                    .with_identifier(format!(
                        "{}:Cancelled",
                        self.namespace.as_deref().unwrap_or("RunMat")
                    ))
                    .with_builtin(self.builtin)
                    .build()
            }
            other => other.into_runtime_error(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::CpuCooperationPolicy;
    use futures::executor::block_on;
    use runmat_execution::cancellation::CancellationSource;
    use std::num::NonZeroU64;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn service_cancelled_is_typed_even_when_invocation_token_is_not_marked() {
        let source = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            || Box::pin(async { Err(ExecutionServiceError::Cancelled) }),
        );
        let mut cooperation = AudioCooperation::with_budget(budget, "audioread");
        let error = block_on(cooperation.completed(1)).expect_err("service cancellation");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert!(cooperation.was_cancelled());
        assert!(!source.token().is_cancelled());
    }
}
