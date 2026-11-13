//! Import-owned work accounting; scheduling and cancellation belong to the
//! invocation execution service, never to this parser.

use super::*;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use std::future::Future;

pub(super) struct ImportCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
}

impl ImportCooperation {
    pub(super) fn new(runtime: Option<&crate::context::RuntimeContext>) -> Self {
        Self {
            budget: runtime.map(crate::context::RuntimeContext::cpu_cooperation),
            namespace: runtime.map(crate::context::RuntimeContext::error_namespace),
        }
    }

    #[cfg(test)]
    pub(super) fn with_budget(budget: CpuCooperationBudget) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".to_string()),
        }
    }

    pub(super) async fn await_io<T, E, F>(
        &mut self,
        future: F,
        input: &input::ImportInput,
    ) -> BuiltinResult<Result<T, E>>
    where
        F: Future<Output = Result<T, E>>,
    {
        input.check_resource()?;
        let result = match self.budget.as_ref() {
            Some(budget) => budget
                .await_cancellable(future)
                .await
                .map_err(|e| self.error(e))?,
            None => future.await,
        };
        if result.is_ok() {
            input.check_resource()?;
        }
        Ok(result)
    }

    pub(super) async fn completed(&mut self, units: usize) -> BuiltinResult<()> {
        let Some(budget) = self.budget.as_mut() else {
            return Ok(());
        };
        budget
            .checkpoint(u64::try_from(units).unwrap_or(u64::MAX))
            .await
            .map_err(|error| self.error(error))
    }

    fn error(&self, error: ExecutionServiceError) -> RuntimeError {
        match error {
            ExecutionServiceError::Cancelled => {
                build_runtime_error("importdata: execution was cancelled")
                    .with_builtin(BUILTIN_NAME)
                    .with_identifier(format!(
                        "{}:Cancelled",
                        self.namespace.as_deref().unwrap_or("RunMat")
                    ))
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
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::num::NonZeroU64;
    use std::rc::Rc;

    #[test]
    fn parser_work_uses_execution_service_turn_and_propagates_cancellation() {
        let source = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).expect("quantum")),
            source.token(),
            || Box::pin(async { Err(ExecutionServiceError::Cancelled) }),
        );
        let mut cooperation = ImportCooperation::with_budget(budget);
        let error = match block_on(text::import_text_data(
            "1 2\n3 4\n",
            None,
            None,
            &mut cooperation,
        )) {
            Ok(_) => panic!("service must cancel during parser work"),
            Err(error) => error,
        };
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
    }

    #[test]
    fn cancelled_import_does_not_poll_provider_read() {
        let source = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::NeverYield,
            source.token(),
            || Box::pin(async { Ok(()) }),
        );
        let input = input::ImportInput::capture(PathBuf::from("/absolute-fixture"))
            .expect("absolute path captures provider without CWD");
        let polled = Rc::new(Cell::new(false));
        source.cancel(CancellationReason::User);
        let mut cooperation = ImportCooperation::with_budget(budget);
        let marker = polled.clone();
        let error = block_on(cooperation.await_io(
            async move {
                marker.set(true);
                Ok::<(), std::io::Error>(())
            },
            &input,
        ))
        .expect_err("cancelled before provider polling");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert!(!polled.get());
    }
}
