//! Exact ownership for a poll-scoped future before queue publication.

use super::{RuntimeContext, RuntimeContextGuard};
use allocator_api2::boxed::Box as FallibleBox;
use runmat_value::ExactHostNumericAllocationError;
use std::future::Future;
use std::pin::Pin;

pub(super) fn pin_in_context<F: Future>(
    context: &RuntimeContext,
    make_future: impl FnOnce() -> F,
) -> Pin<Box<F>> {
    let _guard = RuntimeContextGuard::enter(context.clone());
    Box::pin(make_future())
}

pub(super) fn try_pin_in_context<F: Future>(
    context: &RuntimeContext,
    make_future: impl FnOnce() -> F,
) -> Result<Pin<Box<F>>, ExactHostNumericAllocationError> {
    let _guard = RuntimeContextGuard::enter(context.clone());
    let allocated = FallibleBox::try_new(make_future())
        .map_err(|_| ExactHostNumericAllocationError::AllocationFailed);
    let allocated = match allocated {
        Ok(allocated) => allocated,
        Err(error) => {
            drain_placement_feedback(context);
            return Err(error);
        }
    };
    let pointer = FallibleBox::into_raw(allocated);
    // SAFETY: allocator-api2's Global allocation uses the global allocator;
    // this transfers the one exact allocation into std::Box exactly once.
    Ok(Box::into_pin(unsafe { Box::from_raw(pointer) }))
}

pub(super) fn drop_in_context<F>(context: &RuntimeContext, future: Pin<Box<F>>) {
    let _guard = RuntimeContextGuard::enter(context.clone());
    drop(future);
    drain_placement_feedback(context);
}

pub(super) fn drop_value_in_context<T>(context: &RuntimeContext, value: T) {
    let _guard = RuntimeContextGuard::enter(context.clone());
    drop(value);
    drain_placement_feedback(context);
}

fn drain_placement_feedback(context: &RuntimeContext) {
    if let Some(placement) = context.service_ports().placement() {
        placement.drain_feedback();
    }
}
