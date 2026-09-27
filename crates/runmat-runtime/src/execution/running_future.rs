use crate::context::RuntimeContext;
use crate::execution::ExecutionServiceError;
use runmat_execution::{CancellationReason, FutureHandle};
use runmat_value::{Value, ValueSequence};

/// Owns exactly the service handle transitioned to Running by begin_await.
/// Cancellation/abandonment never drains unrelated futures or sibling scopes.
#[must_use]
pub struct RunningFutureLease {
    runtime: RuntimeContext,
    handle: FutureHandle,
    complete: bool,
}

impl RunningFutureLease {
    /// Take ownership only after this service returns `AwaitAction::ExecuteFuture`.
    pub fn new(runtime: RuntimeContext, handle: FutureHandle) -> Self {
        Self {
            runtime,
            handle,
            complete: false,
        }
    }

    pub fn complete(
        &mut self,
        result: Result<ValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.runtime
            .execution()
            .complete_future(&self.handle, result)?;
        self.complete = true;
        Ok(())
    }
}

impl Drop for RunningFutureLease {
    fn drop(&mut self) {
        if !self.complete {
            let _context = self.runtime.enter();
            let reason = self
                .runtime
                .cancellation()
                .reason()
                .unwrap_or(CancellationReason::ParentScope);
            // Drop cannot report an error. This is the exact live handle from
            // this service; a backend rejection must not trigger a wider drain
            // or a fabricated successful completion.
            let _ = self
                .runtime
                .execution()
                .cancel(&Value::Future(self.handle.clone()), reason);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{
        AwaitAction, DeferredCall, DeferredInvocation, RuntimeExecutionService,
        RuntimeExecutionServices,
    };
    use std::rc::Rc;

    fn deferred() -> DeferredCall {
        DeferredCall {
            invocation: DeferredInvocation::Callable(
                crate::call::descriptor::CallableDescriptor::resolved(
                    runmat_types::CallableIdentity::BoundFunction(runmat_types::FunctionId(1)),
                    vec![],
                    0,
                    runmat_types::CallableFallbackPolicy::None,
                    crate::call::descriptor::CallableCallKind::Direct,
                ),
            ),
            retry: runmat_execution::RetryPolicy::Never,
            program_revision: None,
            capabilities: Default::default(),
            program: None,
        }
    }

    #[test]
    fn abandoned_running_handle_is_cancelled_without_draining_sibling() {
        let service = Rc::new(RuntimeExecutionService::new());
        let runtime = RuntimeContext::new(service.clone());
        let owned = service.create_future(deferred()).unwrap();
        let sibling = service.create_future(deferred()).unwrap();
        assert!(matches!(
            service.begin_await(Value::Future(owned.clone())).unwrap(),
            AwaitAction::ExecuteFuture { .. }
        ));
        let running = RunningFutureLease::new(runtime.clone(), owned.clone());
        drop(running);
        assert!(matches!(
            service.begin_await(Value::Future(owned)),
            Err(ExecutionServiceError::Cancelled)
        ));
        assert!(matches!(
            service.begin_await(Value::Future(sibling.clone())).unwrap(),
            AwaitAction::ExecuteFuture { .. }
        ));
        let mut sibling = RunningFutureLease::new(runtime, sibling);
        sibling.complete(Ok(ValueSequence::empty())).unwrap();
        let handle = sibling.handle.clone();
        drop(sibling);
        assert!(matches!(
            service.begin_await(Value::Future(handle)).unwrap(),
            AwaitAction::Completed(_)
        ));
    }

    #[test]
    fn completed_failure_is_not_replaced_with_cancellation_on_drop() {
        let service = Rc::new(RuntimeExecutionService::new());
        let runtime = RuntimeContext::new(service.clone());
        let handle = service.create_future(deferred()).unwrap();
        assert!(matches!(
            service.begin_await(Value::Future(handle.clone())).unwrap(),
            AwaitAction::ExecuteFuture { .. }
        ));
        let mut running = RunningFutureLease::new(runtime, handle.clone());
        running
            .complete(Err(ExecutionServiceError::Failed("body failure".into())))
            .unwrap();
        drop(running);
        let Err(ExecutionServiceError::Failed(message)) =
            service.begin_await(Value::Future(handle))
        else {
            panic!("completion must retain the language failure");
        };
        assert_eq!(message, "body failure");
    }
}
