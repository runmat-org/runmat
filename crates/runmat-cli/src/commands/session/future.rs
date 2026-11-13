//! Same-lane future construction boundary for CLI command execution.
use std::future::Future;
use std::pin::Pin;

// Keep large child-future construction off the caller's polling frame as well
// as its retained state. The factory returns before the future is polled;
// polling and cancellation/drop still happen on the original execution lane.
#[inline(never)]
pub(crate) fn boxed_command<F: Future>(make_future: impl FnOnce() -> F) -> Pin<Box<F>> {
    Box::pin(make_future())
}
