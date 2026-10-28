//! Invocation-scoped owned request/reply callable, driven by its caller.
//!
//! The registry contains no solver references. Only `drive` borrows a handler,
//! so a user-retained capability can never retain or access a caller's stack.
mod port;
mod origin;
mod child;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod session_tests;
use super::{RootedValueSequence, RuntimeContinuationFuture};
use crate::{call::RootedFunctionArguments, context::RuntimeContext, BuiltinResult, RuntimeError};
use futures::{
    channel::oneshot,
    future::{select, Either},
    StreamExt,
};
use runmat_execution::cancellation::CancellationToken;
use runmat_value::ScopedFunctionHandle;
use std::{future::Future, ops::ControlFlow};

pub(crate) struct Request {
    pub arguments: RootedFunctionArguments,
    pub requested_outputs: usize,
    pub function_input_callsite: Option<crate::callsite::CallsiteInfo>,
}
#[derive(Clone, Copy)]
enum Failure {
    Expired,
    Foreign,
    Busy,
    Unavailable,
    Closed,
}
fn failure(kind: Failure) -> RuntimeError {
    let (identifier, message) = match kind {
        Failure::Expired => (
            "RunMat:ScopedCallable:Expired",
            "the scoped callable's invocation has retired",
        ),
        Failure::Foreign => (
            "RunMat:ScopedCallable:Foreign",
            "the scoped callable belongs to another execution owner or cancellation generation",
        ),
        Failure::Busy => (
            "RunMat:ScopedCallable:Busy",
            "the scoped callable cannot recursively enter its active evaluator",
        ),
        Failure::Unavailable => (
            "RunMat:ScopedCallable:Unavailable",
            "the scoped callable has no port on this execution lane",
        ),
        Failure::Closed => (
            "RunMat:ScopedCallable:Closed",
            "the scoped callable request retired without a reply",
        ),
    };
    crate::build_runtime_error(message)
        .with_identifier(identifier)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}
/// Revalidate wrappers before they can return cached data without invoking.
pub(crate) fn validate_target(
    runtime: Option<&RuntimeContext>,
    handle: &ScopedFunctionHandle,
) -> BuiltinResult<()> {
    port::lookup(runtime, handle).map(|_| ())
}

/// Retain a live capability during child preparation, while its owner is
/// parked. This confers no permission to invoke or read a cached result:
/// those paths still require an actively driven target via validate_target.
pub(crate) fn validate_retention(
    runtime: Option<&RuntimeContext>,
    handle: &ScopedFunctionHandle,
) -> BuiltinResult<()> {
    port::validate_retention(runtime, handle)
}

fn check_original_cancellation(
    runtime: Option<&RuntimeContext>,
    token: Option<&CancellationToken>,
) -> BuiltinResult<()> {
    if token.is_some_and(CancellationToken::is_cancelled) {
        return Err(super::ExecutionServiceError::Cancelled.into_runtime_error_in_context(runtime));
    }
    Ok(())
}

/// The concrete request future owns original argument roots, not Values copied
/// out of them. Its continuation is admitted before any message is enqueued.
pub(crate) fn invoke(
    runtime: Option<&RuntimeContext>,
    handle: ScopedFunctionHandle,
    arguments: RootedFunctionArguments,
    requested_outputs: usize,
    function_input_callsite: Option<crate::callsite::CallsiteInfo>,
) -> BuiltinResult<RuntimeContinuationFuture<'static, RootedValueSequence>> {
    let port = port::lookup(runtime, &handle)?;
    let original = port.origin.runtime().cloned();
    let make = move || async move {
        port.validate(port.origin.runtime(), &handle)?;
        let (reply, returned) = oneshot::channel();
        port.sender
            .unbounded_send(port::Pending {
                call: Request {
                    arguments,
                    requested_outputs,
                    function_input_callsite,
                },
                reply,
            })
            .map_err(|_| failure(Failure::Closed))?;
        let result = returned.await.map_err(|_| failure(Failure::Closed))?;
        let result = match result {
            Ok(result) => result,
            Err(error) => return Err(error),
        };
        // A reply cannot revive an expired capability or move completion to
        // a later invocation generation.
        if !handle.is_live() {
            return Err(failure(Failure::Expired));
        }
        Ok(result)
    };
    match original.as_ref() {
        Some(original) => super::admit_executable_sequence_with(original, make),
        None => super::admit_optional_hostcall_with(
            None,
            |source| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    None,
                    "RunMat:ScopedCallable:Allocation",
                    "scoped callable continuation",
                    source,
                )
            },
            make,
        ),
    }
}

/// Run a real child invocation while servicing its nested calls using one
/// borrowed original handler. No handler/future reference is stored in Value,
/// a static registry, a worker payload, or an independently scheduled task.
/// A domain stop drops the child and closes the scope rather than exposing a
/// catchable numeric-budget exception and permitting extra user work.
/// Ordinary handler errors are instead moved to the awaiting child, so its
/// normal language try/catch semantics remain intact. An uncaught error then
/// leaves through that child's real completion without cloning or replay.
pub(crate) async fn drive<F, S>(
    runtime: Option<&RuntimeContext>,
    token: Option<&CancellationToken>,
    make_child: impl FnOnce(ScopedFunctionHandle, Option<RuntimeContext>) -> BuiltinResult<F>,
    handler: impl AsyncFnMut(Request) -> BuiltinResult<ControlFlow<S, RootedValueSequence>>,
) -> BuiltinResult<ControlFlow<S, RootedValueSequence>>
where
    F: Future<Output = BuiltinResult<RootedValueSequence>>,
{
    let mut scope = port::Registration::new(runtime, token)?;
    drive_registered(&mut scope, port::AfterChild::Retire, make_child, handler).await
}

/// One invocation-owned callable capability across successive user events.
/// The registry retains no handler or borrowed evaluator. A parked capability
/// cannot run until this same owner explicitly drives a real child invocation.
pub(crate) struct Scope {
    registration: port::Registration,
}
impl Scope {
    pub(crate) fn new(
        runtime: Option<&RuntimeContext>,
        token: Option<&CancellationToken>,
    ) -> BuiltinResult<Self> {
        Ok(Self {
            registration: port::Registration::new(runtime, token)?,
        })
    }
    pub(crate) fn handle(&self) -> ScopedFunctionHandle {
        self.registration.port.handle.clone()
    }
    pub(crate) async fn run<F, S, T>(
        &mut self,
        make_child: impl FnOnce(ScopedFunctionHandle, Option<RuntimeContext>) -> BuiltinResult<F>,
        handler: impl AsyncFnMut(Request) -> BuiltinResult<ControlFlow<S, RootedValueSequence>>,
    ) -> BuiltinResult<ControlFlow<S, T>>
    where
        F: Future<Output = BuiltinResult<T>>,
    {
        check_original_cancellation(
            self.registration.port.origin.runtime(),
            self.registration.port.origin.token(),
        )?;
        // A previous run future may have been dropped before its epilogue.
        // Never service a queued old await as work belonging to this event.
        self.discard_pending();
        let result = drive_registered(
            &mut self.registration,
            port::AfterChild::Park,
            make_child,
            handler,
        )
        .await;
        self.discard_pending();
        result
    }
    fn discard_pending(&mut self) {
        // Roots remain owned until this drain (or registration drop); they
        // are never certified retired merely because the port is parked.
        while let Ok(pending) = self.registration.receiver.try_recv() {
            let _ = pending.reply.send(Err(failure(Failure::Unavailable)));
        }
    }
}

async fn drive_registered<F, S, T>(
    scope: &mut port::Registration,
    after: port::AfterChild,
    make_child: impl FnOnce(ScopedFunctionHandle, Option<RuntimeContext>) -> BuiltinResult<F>,
    mut handler: impl AsyncFnMut(Request) -> BuiltinResult<ControlFlow<S, RootedValueSequence>>,
) -> BuiltinResult<ControlFlow<S, T>>
where
    F: Future<Output = BuiltinResult<T>>,
{
    check_original_cancellation(scope.port.origin.runtime(), scope.port.origin.token())?;
    if scope.port.driving.replace(true) {
        return Err(failure(Failure::Busy));
    }
    let entry_exit = port::ChildExit::new(&scope.port, after);
    let child = scope.port.origin.construct(|| {
        make_child(
            scope.port.handle.clone(),
            scope.port.origin.runtime().cloned(),
        )
    })?;
    // ContextFuture destroys a completed inner future in its Ready poll.
    // Revoke before that destruction, not only when this driver unwinds.
    let child = child::Child::new(child, port::ChildExit::new(&scope.port, after));
    let child = scope.port.origin.scope(child);
    let mut child = std::pin::pin!(child);
    let _retirement = entry_exit;
    loop {
        let waiting = select(child.as_mut(), scope.receiver.next());
        let next = match scope.port.origin.token() {
            Some(token) => super::cooperation::await_original_cancellation(token, waiting)
                .await
                .map_err(|error| {
                    error.into_runtime_error_in_context(scope.port.origin.runtime())
                })?,
            None => waiting.await,
        };
        let pending = match next {
            Either::Left((result, _)) => return result.map(ControlFlow::Continue),
            Either::Right((Some(pending), _)) => pending,
            Either::Right((None, _)) => return Err(failure(Failure::Closed)),
        };
        if pending.reply.is_canceled() {
            continue;
        }
        let _serving = port::Serving::enter(&scope.port)?;
        let mut call = pending.call;
        let callsite = call.function_input_callsite.take();
        let handling = {
            let _guard = scope.port.origin.enter();
            handler(call)
        };
        let handling = scope
            .port
            .origin
            .scope(crate::callsite::with_function_input_callsite(
                callsite, handling,
            ));
        let result = match scope.port.origin.token() {
            Some(token) => super::cooperation::await_original_cancellation(token, handling)
                .await
                .map_err(|error| {
                    error.into_runtime_error_in_context(scope.port.origin.runtime())
                })?,
            None => handling.await,
        };
        let result = result.and_then(|result| result);
        match result {
            Ok(ControlFlow::Break(stop)) => return Ok(ControlFlow::Break(stop)),
            Ok(ControlFlow::Continue(result)) => {
                // If the child dropped this particular await, its reply owns
                // and retires the result normally; nothing is republished.
                let _ = pending.reply.send(Ok(result));
            }
            Err(error) => {
                let _ = pending.reply.send(Err(error));
            }
        }
    }
}
