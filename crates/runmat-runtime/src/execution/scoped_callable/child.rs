//! Retire before the canonical poll-scope owner destroys a completed child.
use super::{port::ChildExit, Future};
use std::{
    pin::Pin,
    task::{Context, Poll},
};

pin_project_lite::pin_project! {
    pub(super) struct Child<F> {
        #[pin]
        future: F,
        retirement: ChildExit,
    }
    impl<F> PinnedDrop for Child<F> {
        fn drop(this: Pin<&mut Self>) {
            // PinnedDrop precedes field destruction for pending children too.
            this.project().retirement.close();
        }
    }
}
impl<F> Child<F> {
    pub fn new(future: F, retirement: ChildExit) -> Self {
        Self { future, retirement }
    }
}
impl<F: Future> Future for Child<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let result = this.future.poll(cx);
        if result.is_ready() {
            this.retirement.close();
        }
        result
    }
}
