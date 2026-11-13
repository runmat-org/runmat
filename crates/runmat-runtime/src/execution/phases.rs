//! Consecutive phases in one admitted future allocation. Cold transition
//! construction must not enlarge the frame that polls the running child.

use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

pin_project_lite::pin_project! {
    #[project = PhaseProjection]
    enum Phase<F, G> {
        First { #[pin] future: F },
        Next { #[pin] future: G },
    }
}

pin_project_lite::pin_project! {
    pub(crate) struct ResultPhases<F, G, M> {
        #[pin]
        phase: Phase<F, G>,
        next: Option<M>,
        finished: bool,
    }
}

/// Run a fallible preparation and then its selected successor without retaining
/// the completed preparation in the successor's poll chain. This composition
/// adds no allocation, task, execution scope or admission authority.
/// Inline construction avoids another by-value future-sized call frame.
#[inline(always)]
pub fn after_result<F, G, M, T, O, E>(first: F, next: M) -> impl Future<Output = Result<O, E>>
where
    F: Future<Output = Result<T, E>>,
    G: Future<Output = Result<O, E>>,
    M: FnOnce(T) -> G,
{
    ResultPhases {
        phase: Phase::First { future: first },
        next: Some(next),
        finished: false,
    }
}

// Pin::set drops the completed first child in place before replacing it. All
// G-sized construction/replacement temporaries stay in this cold frame, not
// in poll while the numerical/validation child is running. No second Box,
// allocation grant, context or scheduler is introduced.
#[inline(never)]
fn start_next<F, G, M, T>(mut phase: Pin<&mut Phase<F, G>>, next: M, value: T)
where
    M: FnOnce(T) -> G,
{
    phase.set(Phase::Next {
        future: next(value),
    });
}

impl<F, G, M, T, O, E> Future for ResultPhases<F, G, M>
where
    F: Future<Output = Result<T, E>>,
    G: Future<Output = Result<O, E>>,
    M: FnOnce(T) -> G,
{
    type Output = Result<O, E>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        loop {
            let mut this = self.as_mut().project();
            assert!(!*this.finished, "result phases polled after completion");
            let result = match this.phase.as_mut().project() {
                PhaseProjection::First { future } => match future.poll(context) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
                    Poll::Ready(Ok(value)) => {
                        let next = this.next.take().expect("unconstructed next phase");
                        // A caught factory/destructor panic must not repoll a
                        // completed predecessor or a partially advanced phase.
                        *this.finished = true;
                        start_next(this.phase.as_mut(), next, value);
                        *this.finished = false;
                        // Both first-child polling and cold construction have
                        // returned. Continue without another scheduler turn.
                        continue;
                    }
                },
                PhaseProjection::Next { future } => future.poll(context),
            };
            if result.is_ready() {
                *this.finished = true;
            }
            return result;
        }
    }
}

#[cfg(test)]
#[path = "phases/tests.rs"]
mod tests;
