use crate::context::RuntimeContext;
use crate::execution::{ExecutionServiceError, RootedValueSequence};
use runmat_execution::CancellationReason;
#[cfg(test)]
use runmat_value::Value;

/// Owns exactly the service handle transitioned to Running by begin_await.
/// Cancellation/abandonment never drains unrelated futures or sibling scopes.
#[must_use]
pub struct RunningFutureLease {
    runtime: RuntimeContext,
    handle: super::FutureExecutionHandle,
    complete: bool,
}

impl RunningFutureLease {
    /// Take ownership only after this service returns `AwaitAction::ExecuteFuture`.
    pub fn new(runtime: RuntimeContext, handle: impl Into<super::FutureExecutionHandle>) -> Self {
        Self {
            runtime,
            handle: handle.into(),
            complete: false,
        }
    }

    pub fn complete(
        &mut self,
        result: Result<RootedValueSequence, ExecutionServiceError>,
    ) -> Result<(), ExecutionServiceError> {
        self.runtime
            .execution()
            .complete_future(self.handle.completion_ref(), result)?;
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
                .cancel(&self.handle.retained_value(), reason);
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
            spawn_safety: runmat_types::SpawnSafetyFact::RequiresIsolation,
            invocation: DeferredInvocation::callable(
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
            environment: runmat_execution::EnvironmentSnapshot::default(),
            console: None,
            capabilities: Default::default(),
            program: crate::execution::DeferredProgram::Local,
        }
    }

    #[test]
    fn abandoned_running_handle_is_cancelled_without_draining_sibling() {
        let service = Rc::new(RuntimeExecutionService::new());
        let runtime = RuntimeContext::new(service.clone());
        let owned =
            runmat_value::SharedFutureHandle::new(service.create_future(deferred()).unwrap());
        let sibling = service.create_future(deferred()).unwrap();
        let AwaitAction::ExecuteFuture { handle, .. } =
            service.begin_await(Value::Future(owned.clone())).unwrap()
        else {
            panic!("original future did not enter caller execution");
        };
        assert!(std::ptr::eq(
            handle.completion_ref().outputs,
            &owned.outputs
        ));
        let running = RunningFutureLease::new(runtime.clone(), handle);
        drop(running);
        assert!(matches!(
            service.begin_await(Value::Future(owned)),
            Err(ExecutionServiceError::Cancelled)
        ));
        assert!(matches!(
            service
                .begin_await(Value::Future(sibling.clone().into()))
                .unwrap(),
            AwaitAction::ExecuteFuture { .. }
        ));
        let mut sibling = RunningFutureLease::new(runtime, sibling);
        sibling
            .complete(Ok(RootedValueSequence::new(
                runmat_value::ValueSequence::empty(),
            )
            .unwrap()))
            .unwrap();
        let handle = sibling.handle.clone();
        drop(sibling);
        assert!(matches!(
            service.begin_await(handle.retained_value()).unwrap(),
            AwaitAction::Completed(_)
        ));
    }

    #[test]
    fn completed_failure_is_not_replaced_with_cancellation_on_drop() {
        let service = Rc::new(RuntimeExecutionService::new());
        let runtime = RuntimeContext::new(service.clone());
        let handle = service.create_future(deferred()).unwrap();
        assert!(matches!(
            service
                .begin_await(Value::Future(handle.clone().into()))
                .unwrap(),
            AwaitAction::ExecuteFuture { .. }
        ));
        let mut running = RunningFutureLease::new(runtime, handle.clone());
        running
            .complete(Err(ExecutionServiceError::Failed("body failure".into())))
            .unwrap();
        drop(running);
        let Err(ExecutionServiceError::Failed(message)) =
            service.begin_await(Value::Future(handle.into()))
        else {
            panic!("completion must retain the language failure");
        };
        assert_eq!(message, "body failure");
    }
}
