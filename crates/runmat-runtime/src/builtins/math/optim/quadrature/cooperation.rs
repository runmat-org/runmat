//! One invocation-owned CPU budget across all adaptive subintervals.

use crate::builtins::common::validation::{self, IntegerBoundaryFailure, PreparedIntegerBoundary};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry};
use runmat_value::Value;

#[derive(Clone, Copy)]
pub(crate) enum QuadratureIdentity {
    Integral,
    Quad,
}

impl QuadratureIdentity {
    const fn name(self) -> &'static str {
        match self {
            Self::Integral => "integral",
            Self::Quad => "quad",
        }
    }
}

pub(crate) struct QuadratureCooperation {
    budget: Option<CpuCooperationBudget>,
    namespace: Option<String>,
    identity: QuadratureIdentity,
}

impl QuadratureCooperation {
    pub(crate) fn new(identity: QuadratureIdentity, runtime: Option<&RuntimeContext>) -> Self {
        Self {
            budget: runtime.map(RuntimeContext::cpu_cooperation),
            namespace: runtime.map(RuntimeContext::error_namespace),
            identity,
        }
    }

    pub(crate) async fn completed(&mut self, units: u64) -> BuiltinResult<()> {
        let Some(budget) = self.budget.as_mut() else {
            return Ok(());
        };
        let result = budget.checkpoint(units).await;
        result.map_err(|source| self.execution_failure(source))
    }

    fn execution_failure(&self, source: ExecutionServiceError) -> crate::RuntimeError {
        match source {
            source @ ExecutionServiceError::Cancelled => {
                let namespace = self.namespace.as_deref().unwrap_or("RunMat");
                build_runtime_error(format!("{}: execution was cancelled", self.identity.name()))
                    .with_identifier(format!("{namespace}:Cancelled"))
                    .with_builtin(self.identity.name())
                    .with_source(source)
                    .with_gpu_gather_retry(GpuGatherRetry::Never)
                    .build()
            }
            other => {
                let typed_source = other.clone();
                let mut failure = other.into_runtime_error();
                failure.context = failure.context.with_builtin(self.identity.name());
                failure.gpu_gather_retry = GpuGatherRetry::Never;
                if failure.source.is_none() {
                    failure.source = Some(Box::new(typed_source));
                }
                failure
            }
        }
    }

    pub(crate) async fn prepare_integer_boundary<'a>(
        &mut self,
        value: &'a Value,
    ) -> BuiltinResult<PreparedIntegerBoundary<'a>> {
        let result =
            validation::prepare_direct_integer_f64_boundary(value, self.budget.as_mut()).await;
        result.map_err(|error| match error {
            IntegerBoundaryFailure::Execution(source) => self.execution_failure(source),
            IntegerBoundaryFailure::Operational(source) => {
                let mut failure = *source;
                failure.context.builtin = Some(self.identity.name().to_string());
                failure.gpu_gather_retry = GpuGatherRetry::Never;
                failure
            }
            other => other.into_runtime_error(),
        })
    }

    #[cfg(test)]
    pub(crate) fn with_budget(identity: QuadratureIdentity, budget: CpuCooperationBudget) -> Self {
        Self {
            budget: Some(budget),
            namespace: Some("RunMat".into()),
            identity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::CpuCooperationPolicy;
    use futures::task::noop_waker;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::future::Future;
    use std::num::NonZeroU64;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    struct PendingTurn(Rc<Cell<bool>>);

    impl Future for PendingTurn {
        type Output = Result<(), ExecutionServiceError>;

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }

    impl Drop for PendingTurn {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    #[test]
    fn pending_turn_cancellation_retains_typed_source_and_retires_turn() {
        let source = CancellationSource::new();
        let dropped = Rc::new(Cell::new(false));
        let witness = dropped.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || Box::pin(PendingTurn(witness.clone())),
        );
        let mut cooperation = QuadratureCooperation::with_budget(QuadratureIdentity::Quad, budget);
        let mut future = Box::pin(cooperation.completed(1));
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
        source.cancel(CancellationReason::User);
        let Poll::Ready(Err(error)) = future.as_mut().poll(&mut context) else {
            panic!("cancelled pending work must fail");
        };
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
        assert!(error.source.is_some());
        drop(future);
        assert!(dropped.get());
    }

    #[test]
    fn turn_failure_and_preexisting_cancel_are_not_retried() {
        let source = CancellationSource::new();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            || Box::pin(async { Err(ExecutionServiceError::Infrastructure("lost turn".into())) }),
        );
        let mut cooperation =
            QuadratureCooperation::with_budget(QuadratureIdentity::Integral, budget);
        let error = futures::executor::block_on(cooperation.completed(1)).unwrap_err();
        assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
        assert_eq!(
            error.identifier(),
            Some("RunMat:parallel:InfrastructureFailure")
        );
        assert!(matches!(
            error
                .source
                .as_ref()
                .and_then(|source| source.downcast_ref::<ExecutionServiceError>()),
            Some(ExecutionServiceError::Infrastructure(message)) if message == "lost turn"
        ));
        source.cancel(CancellationReason::User);
        let cancelled = futures::executor::block_on(cooperation.completed(0)).unwrap_err();
        assert_eq!(cancelled.identifier(), Some("RunMat:Cancelled"));
        assert!(cancelled.source.is_some());
    }

    #[test]
    fn abandoned_pending_turn_drops_without_completion() {
        let source = CancellationSource::new();
        let dropped = Rc::new(Cell::new(false));
        let witness = dropped.clone();
        let budget = CpuCooperationBudget::with_test_turn(
            CpuCooperationPolicy::WorkQuantum(NonZeroU64::new(1).unwrap()),
            source.token(),
            move || Box::pin(PendingTurn(witness.clone())),
        );
        let mut cooperation =
            QuadratureCooperation::with_budget(QuadratureIdentity::Integral, budget);
        let mut future = Box::pin(cooperation.completed(1));
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(future.as_mut().poll(&mut context).is_pending());
        drop(future);
        assert!(dropped.get());
        assert!(!source.token().is_cancelled());
    }
}
