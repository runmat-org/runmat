//! One admitted heap boundary for a rooted runtime continuation.
//!
//! Builtin bodies and dynamically selected callables use the same concrete
//! future owner. Their domains supply the error presentation; this owner only
//! enforces the captured cancellation/resource authority and root lifetime.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::{RuntimeCancellationContext, RuntimeContext};
use crate::RuntimeError;

use super::{RootedValue, RootedValueSequence};

/// An erased continuation returned by selected executable and VM admission.
pub type RuntimeContinuationFuture<'a, O> =
    Pin<Box<dyn Future<Output = Result<O, RuntimeError>> + 'a>>;

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::RootedValue {}
    impl Sealed for super::RootedValueSequence {}
    impl Sealed for Option<super::RootedValue> {}
    // VM stack mutations keep their Values in the instruction owner's rooted
    // stack; the continuation itself returns no unrooted Value.
    impl Sealed for () {}
}

/// Only canonical rooted completion carriers may leave this boundary.
pub(crate) trait RootedCompletion: sealed::Sealed {}
impl RootedCompletion for RootedValue {}
impl RootedCompletion for RootedValueSequence {}
impl RootedCompletion for Option<RootedValue> {}
impl RootedCompletion for () {}

pub(crate) type ControlErrorMapper =
    fn(Option<&RuntimeCancellationContext>, &'static str, HostAllocationError) -> RuntimeError;

pin_project_lite::pin_project! {
    /// The rooted result is produced before the input future can retire. Field
    /// order drops the concrete future before its host allocation claim.
    struct AdmittedRootedFuture<F, O> {
        #[pin]
        future: F,
        claim: Option<HostAllocationReceipt>,
        cancellation: Option<RuntimeCancellationContext>,
        label: &'static str,
        map_error: ControlErrorMapper,
        completed: bool,
        _result: PhantomData<fn() -> O>,
    }
}

impl<F, O> Future for AdmittedRootedFuture<F, O>
where
    F: Future<Output = Result<O, RuntimeError>>,
    O: RootedCompletion,
{
    type Output = Result<O, RuntimeError>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        assert!(
            !*this.completed,
            "rooted continuation polled after completion"
        );
        if let Err(error) = checkpoint(this.claim.as_ref(), this.cancellation.as_ref()) {
            *this.completed = true;
            return Poll::Ready(Err((this.map_error)(
                this.cancellation.as_ref(),
                this.label,
                error,
            )));
        }
        match this.future.poll(context) {
            Poll::Ready(Ok(output)) => {
                // A rooted output cannot cross a cancellation or revocation
                // that arrived during the final poll.
                let checked = checkpoint(this.claim.as_ref(), this.cancellation.as_ref());
                *this.completed = true;
                match checked {
                    Ok(()) => Poll::Ready(Ok(output)),
                    Err(error) => Poll::Ready(Err((this.map_error)(
                        this.cancellation.as_ref(),
                        this.label,
                        error,
                    ))),
                }
            }
            Poll::Ready(Err(error)) => {
                *this.completed = true;
                Poll::Ready(Err(error))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Admit the exact target-local owner layout before the first poll or heap
/// allocation. Constructing an async future does not execute its body.
pub(crate) fn admit_rooted_future<'a, F, O>(
    runtime: Option<&RuntimeContext>,
    label: &'static str,
    map_error: ControlErrorMapper,
    future: F,
) -> Pin<Box<dyn Future<Output = Result<O, RuntimeError>> + 'a>>
where
    F: Future<Output = Result<O, RuntimeError>> + 'a,
    O: RootedCompletion + 'a,
{
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    let admission = admit_layout::<F, O>(runtime);
    match admission {
        Ok(claim) => Box::pin(AdmittedRootedFuture {
            future,
            claim,
            cancellation,
            label,
            map_error,
            completed: false,
            _result: PhantomData,
        }),
        Err(error) => {
            drop(future);
            Box::pin(std::future::ready(Err(map_error(
                cancellation.as_ref(),
                label,
                error,
            ))))
        }
    }
}

/// An owned-context variant for a dynamic callback. The same context is used
/// for admission and moved into the selected dispatch future; the callback
/// body is not constructed when admission fails.
pub(crate) fn admit_rooted_future_with<'a, F, O>(
    runtime: Option<RuntimeContext>,
    label: &'static str,
    map_error: ControlErrorMapper,
    make_future: impl FnOnce(Option<RuntimeContext>) -> F,
) -> Pin<Box<dyn Future<Output = Result<O, RuntimeError>> + 'a>>
where
    F: Future<Output = Result<O, RuntimeError>> + 'a,
    O: RootedCompletion + 'a,
{
    let cancellation = runtime.as_ref().map(RuntimeContext::cancellation_context);
    match admit_layout::<F, O>(runtime.as_ref()) {
        Ok(claim) => Box::pin(AdmittedRootedFuture {
            future: make_future(runtime),
            claim,
            cancellation,
            label,
            map_error,
            completed: false,
            _result: PhantomData,
        }),
        Err(error) => Box::pin(std::future::ready(Err(map_error(
            cancellation.as_ref(),
            label,
            error,
        )))),
    }
}

fn admit_layout<F, O>(
    runtime: Option<&RuntimeContext>,
) -> Result<Option<HostAllocationReceipt>, HostAllocationError>
where
    F: Future<Output = Result<O, RuntimeError>>,
    O: RootedCompletion,
{
    rooted_future_layout::<F, O>()
        .and_then(|bytes| HostAllocationPlan::new(bytes, 0))
        .and_then(|plan| HostAllocationReceipt::admit(runtime, plan))
}

pub(crate) fn rooted_future_layout<F, O>() -> Result<u64, HostAllocationError>
where
    F: Future<Output = Result<O, RuntimeError>>,
    O: RootedCompletion,
{
    u64::try_from(std::mem::size_of::<AdmittedRootedFuture<F, O>>())
        .map_err(|_| HostAllocationError::Overflow)
}

fn checkpoint(
    claim: Option<&HostAllocationReceipt>,
    cancellation: Option<&RuntimeCancellationContext>,
) -> Result<(), HostAllocationError> {
    if cancellation.is_some_and(RuntimeCancellationContext::is_cancelled) {
        return Err(HostAllocationError::Cancelled);
    }
    claim.map_or(Ok(()), HostAllocationReceipt::checkpoint_captured)
}

/// Core's selected executable target uses the same rooted, admitted future
/// owner as dynamic callbacks. Only the diagnostic presentation differs from
/// a builtin; its cancellation and resource causes remain typed.
pub fn admit_executable_sequence<'a, F>(
    runtime: &RuntimeContext,
    future: F,
) -> Result<RuntimeContinuationFuture<'a, RootedValueSequence>, RuntimeError>
where
    F: Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a,
{
    admit_selected_future(runtime, "executable", future)
}

/// Admit one selected VM stack mutation. Its existing stack owner holds all
/// operands/results as GC roots; the admitted future returns only unit.
pub fn admit_stack_mutation<'a, F>(
    runtime: &RuntimeContext,
    future: F,
) -> Result<RuntimeContinuationFuture<'a, ()>, RuntimeError>
where
    F: Future<Output = Result<(), RuntimeError>> + 'a,
{
    admit_selected_future(runtime, "arithmetic", future)
}

fn admit_selected_future<'a, F, O>(
    runtime: &RuntimeContext,
    label: &'static str,
    future: F,
) -> Result<RuntimeContinuationFuture<'a, O>, RuntimeError>
where
    F: Future<Output = Result<O, RuntimeError>> + 'a,
    O: RootedCompletion + 'a,
{
    let cancellation = Some(runtime.cancellation_context());
    let claim = admit_layout::<F, O>(Some(runtime))
        .map_err(|error| semantic_control_error(cancellation.as_ref(), label, error))?;
    Ok(Box::pin(AdmittedRootedFuture {
        future,
        claim,
        cancellation,
        label,
        map_error: semantic_control_error,
        completed: false,
        _result: PhantomData,
    }))
}

fn semantic_control_error(
    cancellation: Option<&RuntimeCancellationContext>,
    label: &'static str,
    source: HostAllocationError,
) -> RuntimeError {
    use crate::context::host_effect::HostEffectAllocationError;
    use crate::execution::ExecutionServiceError;
    match source {
        HostAllocationError::Cancelled
        | HostAllocationError::Resource(HostEffectAllocationError::Cancelled) => {
            let mut error = ExecutionServiceError::Cancelled.into_runtime_error();
            if let Some(cancellation) = cancellation {
                error.identifier = Some(format!("{}:Cancelled", cancellation.error_namespace()));
            }
            error
        }
        HostAllocationError::StaleAuthority
        | HostAllocationError::MissingManagedOwner
        | HostAllocationError::Resource(HostEffectAllocationError::Revoked) => {
            ExecutionServiceError::Infrastructure(format!(
                "managed {label} continuation authority changed"
            ))
            .into_runtime_error()
        }
        _ => crate::runtime_error::semantic_error_with_source(
            "ContinuationResource",
            format!("{label} continuation allocation failed"),
            source,
        ),
    }
}
