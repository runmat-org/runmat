use runmat_execution::{CancellationReason, FutureHandle};
use runmat_runtime::context::RuntimeContext;
use runmat_runtime::execution::ExecutionServiceError;
use runmat_value::{Value, ValueSequence};

/// Owns exactly the service handle transitioned to Running by begin_await.
/// Cancellation/abandonment never drains unrelated futures or sibling scopes.
pub(super) struct RunningFuture {
    runtime: RuntimeContext,
    handle: FutureHandle,
    complete: bool,
}

impl RunningFuture {
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

impl Drop for RunningFuture {
    fn drop(&mut self) {
        if !self.complete {
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
    use runmat_runtime::execution::{
        AwaitAction, DeferredCall, DeferredInvocation, RuntimeExecutionService,
        RuntimeExecutionServices,
    };
    use std::rc::Rc;

    fn deferred() -> DeferredCall {
        DeferredCall {
            invocation: DeferredInvocation::Callable(
                runmat_runtime::call::descriptor::CallableDescriptor::resolved(
                    runmat_types::CallableIdentity::BoundFunction(runmat_types::FunctionId(1)),
                    vec![],
                    0,
                    runmat_types::CallableFallbackPolicy::None,
                    runmat_runtime::call::descriptor::CallableCallKind::Direct,
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
        let running = RunningFuture::new(runtime.clone(), owned.clone());
        drop(running);
        assert!(matches!(
            service.begin_await(Value::Future(owned)),
            Err(ExecutionServiceError::Cancelled)
        ));
        assert!(matches!(
            service.begin_await(Value::Future(sibling.clone())).unwrap(),
            AwaitAction::ExecuteFuture { .. }
        ));
        let mut sibling = RunningFuture::new(runtime, sibling);
        sibling.complete(Ok(ValueSequence::empty())).unwrap();
        let handle = sibling.handle.clone();
        drop(sibling);
        assert!(matches!(
            service.begin_await(Value::Future(handle)).unwrap(),
            AwaitAction::Completed(_)
        ));
    }
}
