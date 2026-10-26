//! Invocation cancellation for externally pending work owned by a call.

use futures::future::{select, Either};
use runmat_execution::cancellation::CancellationToken;
use std::future::Future;

use super::super::ExecutionServiceError;

pub(in crate::execution) async fn run<F: Future>(
    cancellation: &CancellationToken,
    future: F,
) -> Result<F::Output, ExecutionServiceError> {
    if cancellation.is_cancelled() {
        return Err(ExecutionServiceError::Cancelled);
    }
    let work = std::pin::pin!(future);
    // Poll the listener first on every wake. A provider may wake its own
    // pending open; if cancellation arrives before that wake is serviced,
    // polling the open first could perform the effect before we notice it.
    let result = match select(cancellation.cancelled(), work).await {
        Either::Left((_reason, _work)) => return Err(ExecutionServiceError::Cancelled),
        Either::Right((result, _listener)) => result,
    };
    if cancellation.is_cancelled() {
        return Err(ExecutionServiceError::Cancelled);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::task::noop_waker_ref;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_execution::CancellationReason;
    use std::cell::Cell;
    use std::pin::Pin;
    use std::rc::Rc;
    use std::task::{Context, Poll};

    struct PendingDrop(Rc<Cell<bool>>);
    impl Future for PendingDrop {
        type Output = ();

        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }
    impl Drop for PendingDrop {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    #[test]
    fn pending_work_is_woken_and_dropped_on_cancellation() {
        let source = CancellationSource::new();
        let dropped = Rc::new(Cell::new(false));
        let token = source.token();
        let mut call = Box::pin(run(&token, PendingDrop(dropped.clone())));
        assert!(matches!(
            call.as_mut()
                .poll(&mut Context::from_waker(noop_waker_ref())),
            Poll::Pending
        ));
        source.cancel(CancellationReason::User);
        assert!(matches!(
            call.as_mut()
                .poll(&mut Context::from_waker(noop_waker_ref())),
            Poll::Ready(Err(ExecutionServiceError::Cancelled))
        ));
        assert!(dropped.get());
    }

    #[test]
    fn cancelled_self_waking_work_is_not_polled_again() {
        struct SelfWakingOpen {
            polls: Rc<Cell<usize>>,
            dropped: Rc<Cell<bool>>,
        }
        impl Future for SelfWakingOpen {
            type Output = ();
            fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
                let polls = self.polls.get() + 1;
                self.polls.set(polls);
                if polls == 1 {
                    context.waker().wake_by_ref();
                    Poll::Pending
                } else {
                    Poll::Ready(())
                }
            }
        }
        impl Drop for SelfWakingOpen {
            fn drop(&mut self) {
                self.dropped.set(true);
            }
        }
        let source = CancellationSource::new();
        let polls = Rc::new(Cell::new(0));
        let dropped = Rc::new(Cell::new(false));
        let token = source.token();
        let mut call = Box::pin(run(
            &token,
            SelfWakingOpen {
                polls: polls.clone(),
                dropped: dropped.clone(),
            },
        ));
        assert!(matches!(
            call.as_mut()
                .poll(&mut Context::from_waker(noop_waker_ref())),
            Poll::Pending
        ));
        assert_eq!(polls.get(), 1);
        source.cancel(CancellationReason::User);
        assert!(matches!(
            call.as_mut()
                .poll(&mut Context::from_waker(noop_waker_ref())),
            Poll::Ready(Err(ExecutionServiceError::Cancelled))
        ));
        assert_eq!(
            polls.get(),
            1,
            "cancel must win before a ready provider effect"
        );
        assert!(dropped.get());
    }

    #[test]
    fn already_cancelled_does_not_poll_work() {
        let source = CancellationSource::new();
        source.cancel(CancellationReason::User);
        let dropped = Rc::new(Cell::new(false));
        let result =
            futures::executor::block_on(run(&source.token(), PendingDrop(dropped.clone())));
        assert!(matches!(result, Err(ExecutionServiceError::Cancelled)));
        assert!(dropped.get());
    }

    #[test]
    fn ready_value_and_original_error_are_returned_without_conversion() {
        let source = CancellationSource::new();
        let token = source.token();
        let ready = futures::executor::block_on(run(&token, async { Ok::<_, Rc<()>>(17) }))
            .expect("invocation not cancelled");
        assert_eq!(ready, Ok(17));

        let original = Rc::new(());
        let returned =
            futures::executor::block_on(run(&token, async { Err::<(), _>(original.clone()) }))
                .expect("invocation not cancelled")
                .expect_err("provider error must remain the provider error");
        assert!(Rc::ptr_eq(&returned, &original));
    }

    #[test]
    fn cancellation_during_ready_poll_drops_value_and_wins_publication() {
        struct DropValue(Rc<Cell<bool>>);
        impl Drop for DropValue {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }
        struct CancelThenReady {
            source: CancellationSource,
            dropped: Rc<Cell<bool>>,
        }
        impl Future for CancelThenReady {
            type Output = DropValue;
            fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
                self.source.cancel(CancellationReason::User);
                Poll::Ready(DropValue(self.dropped.clone()))
            }
        }
        let source = CancellationSource::new();
        let dropped = Rc::new(Cell::new(false));
        let result = futures::executor::block_on(run(
            &source.token(),
            CancelThenReady {
                source: source.clone(),
                dropped: dropped.clone(),
            },
        ));
        assert!(matches!(result, Err(ExecutionServiceError::Cancelled)));
        assert!(
            dropped.get(),
            "ready value must not escape after cancellation"
        );
    }

    #[test]
    fn unpolled_await_drops_work_without_polling_or_registering() {
        struct CountPolls {
            polled: Rc<Cell<bool>>,
            dropped: Rc<Cell<bool>>,
        }
        impl Future for CountPolls {
            type Output = ();
            fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
                self.polled.set(true);
                Poll::Pending
            }
        }
        impl Drop for CountPolls {
            fn drop(&mut self) {
                self.dropped.set(true);
            }
        }
        let source = CancellationSource::new();
        let token = source.token();
        let polled = Rc::new(Cell::new(false));
        let dropped = Rc::new(Cell::new(false));
        drop(run(
            &token,
            CountPolls {
                polled: polled.clone(),
                dropped: dropped.clone(),
            },
        ));
        assert!(!polled.get());
        assert!(dropped.get());
    }
}
