//! One pinned typed request whose inner future survives its ready poll.
//! The queue drops this envelope after descendants, before publishing the
//! result held by the shared completion cell.

use super::CompletionOwner;
use runmat_runtime::RuntimeError;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

pin_project_lite::pin_project! {
    pub(super) struct TypedQueuedRequest<F, T> {
        #[pin]
        future: F,
        completion: CompletionOwner<T>,
    }
}

impl<F, T> TypedQueuedRequest<F, T> {
    pub(super) fn new(future: F, completion: CompletionOwner<T>) -> Self {
        Self { future, completion }
    }
}

impl<F, T> Future for TypedQueuedRequest<F, T>
where
    F: Future<Output = Result<T, RuntimeError>>,
{
    type Output = ();

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        match this.future.poll(context) {
            Poll::Ready(result) => {
                *this.completion.get().result.borrow_mut() = Some(result);
                Poll::Ready(())
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
