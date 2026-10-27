//! Allocation-free iteration whose completed step retires before its successor.

use std::{
    future::Future,
    ops::ControlFlow,
    pin::Pin,
    task::{Context, Poll},
};

pin_project_lite::pin_project! {
    struct ResultLoop<I, F, M> {
        #[pin]
        future: Option<F>,
        input: Option<I>,
        make: M,
        finished: bool,
    }
}

#[inline(always)]
pub(crate) fn loop_result<I, F, M, O, E>(input: I, make: M) -> impl Future<Output = Result<O, E>>
where
    M: FnMut(I) -> F,
    F: Future<Output = Result<ControlFlow<O, I>, E>>,
{
    ResultLoop {
        future: None,
        input: Some(input),
        make,
        finished: false,
    }
}

// Keep future-sized construction temporaries out of the active child's poll
// chain. Pin::set retires the completed child before calling the next factory.
#[inline(never)]
fn start<I, F, M>(mut future: Pin<&mut Option<F>>, make: &mut M, input: I)
where
    M: FnMut(I) -> F,
{
    future.set(None);
    future.set(Some(make(input)));
}

impl<I, F, M, O, E> Future for ResultLoop<I, F, M>
where
    M: FnMut(I) -> F,
    F: Future<Output = Result<ControlFlow<O, I>, E>>,
{
    type Output = Result<O, E>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            let mut this = self.as_mut().project();
            assert!(!*this.finished, "result loop polled after completion");
            if let Some(input) = this.input.take() {
                start(this.future.as_mut(), this.make, input);
            }
            let result = match this
                .future
                .as_mut()
                .as_pin_mut()
                .expect("active result-loop step")
                .poll(context)
            {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(result) => result,
            };
            match result {
                Ok(ControlFlow::Continue(input)) => {
                    start(this.future, this.make, input);
                }
                Ok(ControlFlow::Break(output)) => {
                    this.future.set(None);
                    *this.finished = true;
                    return Poll::Ready(Ok(output));
                }
                Err(error) => {
                    this.future.set(None);
                    *this.finished = true;
                    return Poll::Ready(Err(error));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
