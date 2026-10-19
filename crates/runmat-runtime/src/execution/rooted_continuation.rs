//! One admitted heap boundary for a rooted runtime continuation.
//!
//! Builtin bodies and dynamically selected callables use the same concrete
//! future owner. Their domains supply the error presentation; this owner only
//! enforces the captured cancellation/resource authority and root lifetime.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::task::{Context, Poll};

use runmat_gc::Trace;

use crate::context::host_allocation::{
    HostAllocationError, HostAllocationPlan, HostAllocationReceipt,
};
use crate::context::{RuntimeCancellationContext, RuntimeContext};
use crate::RuntimeError;

use super::{RootedValue, RootedValueSequence};

/// An erased continuation returned by selected executable and VM admission.
/// Runtime-owned entry points retain `RuntimeError` by default, while callers
/// with a stronger domain error may preserve it through the same admitted
/// continuation owner.
pub type RuntimeContinuationFuture<'a, O, E = RuntimeError> =
    Pin<Box<dyn Future<Output = Result<O, E>> + 'a>>;

mod sealed {
    pub trait Sealed {}
    impl<T: runmat_gc::Trace> Sealed for super::super::RootedResult<T> {}
    impl Sealed for Option<super::RootedValue> {}
    // VM stack mutations keep their Values in the instruction owner's rooted
    // stack; the continuation itself returns no unrooted Value.
    impl Sealed for () {}
}

/// Only canonical rooted completion carriers may leave this boundary.
pub(crate) trait RootedCompletion: sealed::Sealed {}
impl<T: Trace> RootedCompletion for super::RootedResult<T> {}
impl RootedCompletion for Option<RootedValue> {}
impl RootedCompletion for () {}

pub(crate) type ContextualControlErrorMapper<E = RuntimeError> =
    fn(Option<&RuntimeCancellationContext>, &'static str, HostAllocationError) -> E;

enum ControlErrorMapper<E> {
    Contextual(ContextualControlErrorMapper<E>),
    Cause(fn(HostAllocationError) -> E),
}

impl<E> ControlErrorMapper<E> {
    fn map(
        &self,
        cancellation: Option<&RuntimeCancellationContext>,
        label: &'static str,
        cause: HostAllocationError,
    ) -> E {
        match self {
            Self::Contextual(map) => map(cancellation, label, cause),
            Self::Cause(map) => map(cause),
        }
    }
}

pin_project_lite::pin_project! {
    /// The result is produced before the input future can retire. Field order
    /// drops the concrete future before its host allocation claim. Rooted
    /// entrypoints separately constrain their result to RootedCompletion.
    struct AdmittedContinuation<F, O, E> {
        #[pin]
        future: F,
        claim: Option<HostAllocationReceipt>,
        cancellation: Option<RuntimeCancellationContext>,
        label: &'static str,
        map_error: ControlErrorMapper<E>,
        completed: bool,
        _result: PhantomData<fn() -> (O, E)>,
    }
}

impl<F, O, E> Future for AdmittedContinuation<F, O, E>
where
    F: Future<Output = Result<O, E>>,
{
    type Output = Result<O, E>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        assert!(
            !*this.completed,
            "admitted continuation polled after completion"
        );
        if let Err(error) = checkpoint(this.claim.as_ref(), this.cancellation.as_ref()) {
            *this.completed = true;
            return Poll::Ready(Err(this.map_error.map(
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
                    Err(error) => Poll::Ready(Err(this.map_error.map(
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
    map_error: ContextualControlErrorMapper,
    future: F,
) -> Pin<Box<dyn Future<Output = Result<O, RuntimeError>> + 'a>>
where
    F: Future<Output = Result<O, RuntimeError>> + 'a,
    O: RootedCompletion + 'a,
{
    let cancellation = runtime.map(RuntimeContext::cancellation_context);
    let admission = admit_layout::<F, O, RuntimeError>(runtime);
    match admission {
        Ok(claim) => Box::pin(AdmittedContinuation {
            future,
            claim,
            cancellation,
            label,
            map_error: ControlErrorMapper::Contextual(map_error),
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
    map_error: ContextualControlErrorMapper,
    make_future: impl FnOnce(Option<RuntimeContext>) -> F,
) -> Pin<Box<dyn Future<Output = Result<O, RuntimeError>> + 'a>>
where
    F: Future<Output = Result<O, RuntimeError>> + 'a,
    O: RootedCompletion + 'a,
{
    let cancellation = runtime.as_ref().map(RuntimeContext::cancellation_context);
    match admit_layout::<F, O, RuntimeError>(runtime.as_ref()) {
        Ok(claim) => Box::pin(AdmittedContinuation {
            future: make_future(runtime),
            claim,
            cancellation,
            label,
            map_error: ControlErrorMapper::Contextual(map_error),
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

fn admit_layout<F, O, E>(
    runtime: Option<&RuntimeContext>,
) -> Result<Option<HostAllocationReceipt>, HostAllocationError>
where
    F: Future<Output = Result<O, E>>,
{
    rooted_future_layout_with_error::<F, O, E>()
        .and_then(|bytes| HostAllocationPlan::new(bytes, 0))
        .and_then(|plan| HostAllocationReceipt::admit(runtime, plan))
}

pub(crate) fn rooted_future_layout<F, O>() -> Result<u64, HostAllocationError>
where
    F: Future<Output = Result<O, RuntimeError>>,
    O: RootedCompletion,
{
    rooted_future_layout_with_error::<F, O, RuntimeError>()
}

fn rooted_future_layout_with_error<F, O, E>() -> Result<u64, HostAllocationError>
where
    F: Future<Output = Result<O, E>>,
{
    u64::try_from(std::mem::size_of::<AdmittedContinuation<F, O, E>>())
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

/// Factory form of executable admission. Construct the concrete executable
/// future only inside the admitted, erased owner so a large selected future
/// does not become part of every caller's poll-frame layout.
#[inline(never)]
pub fn admit_executable_sequence_with<'a, F>(
    runtime: &RuntimeContext,
    make_future: impl FnOnce() -> F,
) -> Result<RuntimeContinuationFuture<'a, RootedValueSequence>, RuntimeError>
where
    F: Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a,
{
    admit_selected_future_with_error_from(
        runtime,
        "executable",
        ControlErrorMapper::Contextual(semantic_control_error),
        make_future,
    )
}

/// Admit a domain-specific canonical rooted result without teaching Runtime
/// about its payload type. The `Trace` bound and sealed `RootedResult` owner
/// preserve the same completion-root invariant as Value sequences.
#[inline(never)]
pub fn admit_rooted_result_with<'a, T, F>(
    runtime: &RuntimeContext,
    make_future: impl FnOnce() -> F,
) -> Result<RuntimeContinuationFuture<'a, super::RootedResult<T>>, RuntimeError>
where
    T: Trace + 'a,
    F: Future<Output = Result<super::RootedResult<T>, RuntimeError>> + 'a,
{
    admit_selected_future_with_error_from(
        runtime,
        "rooted execution",
        ControlErrorMapper::Contextual(semantic_control_error),
        make_future,
    )
}

/// Admit a workspace-aware source continuation before its concrete future is
/// erased. Dynamic evaluation borrows its selected arguments and executable
/// authority, so the returned owner retains that lifetime without a second
/// raw `Box::pin` boundary.
pub fn admit_dynamic_workspace_sequence<'a, F>(
    runtime: &RuntimeContext,
    future: F,
) -> Result<RuntimeContinuationFuture<'a, RootedValueSequence>, RuntimeError>
where
    F: Future<Output = Result<RootedValueSequence, RuntimeError>> + 'a,
{
    admit_selected_future(runtime, "dynamic workspace source", future)
}

/// Admit a selected executable continuation while preserving its domain error
/// type. Admission, cancellation, and authority failures are mapped before
/// they cross the boundary; errors produced by the executable are returned
/// unchanged.
pub fn admit_executable_sequence_with_error<'a, F, E>(
    runtime: &RuntimeContext,
    map_error: fn(HostAllocationError) -> E,
    future: F,
) -> Result<RuntimeContinuationFuture<'a, RootedValueSequence, E>, E>
where
    F: Future<Output = Result<RootedValueSequence, E>> + 'a,
    E: 'a,
{
    admit_selected_future_with_error(
        runtime,
        "executable",
        ControlErrorMapper::Cause(map_error),
        future,
    )
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

/// Admit a domain operation that completes without returning language Values.
/// The small factory keeps the concrete future out of the
/// caller's poll frame; the exact layout is admitted before constructing it.
/// This shares the rooted continuation's allocation/cancellation owner, but
/// the caller remains responsible for roots and retained allocations inside
/// its domain artifact.
#[inline(never)]
pub fn admit_domain_continuation_with<'a, F>(
    runtime: &RuntimeContext,
    make_future: impl FnOnce() -> F,
) -> Result<RuntimeContinuationFuture<'a, ()>, RuntimeError>
where
    F: Future<Output = Result<(), RuntimeError>> + 'a,
{
    admit_selected_future_with_error_from(
        runtime,
        "domain continuation",
        ControlErrorMapper::Contextual(semantic_control_error),
        make_future,
    )
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
    admit_selected_future_with_error(
        runtime,
        label,
        ControlErrorMapper::Contextual(semantic_control_error),
        future,
    )
}

fn admit_selected_future_with_error<'a, F, O, E>(
    runtime: &RuntimeContext,
    label: &'static str,
    map_error: ControlErrorMapper<E>,
    future: F,
) -> Result<RuntimeContinuationFuture<'a, O, E>, E>
where
    F: Future<Output = Result<O, E>> + 'a,
    O: 'a,
    E: 'a,
{
    admit_selected_future_with_error_from(runtime, label, map_error, || future)
}

#[inline(never)]
fn admit_selected_future_with_error_from<'a, F, O, E>(
    runtime: &RuntimeContext,
    label: &'static str,
    map_error: ControlErrorMapper<E>,
    make_future: impl FnOnce() -> F,
) -> Result<RuntimeContinuationFuture<'a, O, E>, E>
where
    F: Future<Output = Result<O, E>> + 'a,
    O: 'a,
    E: 'a,
{
    let cancellation = Some(runtime.cancellation_context());
    let claim = admit_layout::<F, O, E>(Some(runtime))
        .map_err(|error| map_error.map(cancellation.as_ref(), label, error))?;
    let future = make_future();
    Ok(Box::pin(AdmittedContinuation {
        future,
        claim,
        cancellation,
        label,
        map_error,
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
