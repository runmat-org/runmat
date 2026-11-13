//! Wakeable cancellation of an execution scope; synchronous execution allocates
//! no cancellation listener. The caller retains terminal VM cleanup ownership.
use runmat_execution::cancellation::CancellationToken;
use std::future::{poll_fn, Future};
use std::pin::pin;
use std::task::Poll;

pub(super) async fn until_cancelled<F: Future>(
    token: &CancellationToken,
    operation: F,
) -> Option<F::Output> {
    let mut operation = pin!(operation);
    let mut listener = None;
    poll_fn(|cx| {
        if token.is_cancelled() {
            return Poll::Ready(None);
        }
        if let Poll::Ready(result) = operation.as_mut().poll(cx) {
            // A provider may observe cancellation and return its own terminal
            // error during this poll. Cancellation is invocation control, not
            // a catchable provider failure, so preserve the VM cancellation
            // contract instead of publishing that operational wrapper.
            return Poll::Ready((!token.is_cancelled()).then_some(result));
        }
        // Registration rechecks cancellation under the authority's lock, so a
        // signal during the operation's first poll cannot be lost.
        let listener = listener.get_or_insert_with(|| token.cancelled());
        match std::pin::Pin::new(listener).poll(cx) {
            Poll::Ready(_) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_execution::{cancellation::CancellationSource, CancellationReason};
    use std::cell::Cell;

    #[test]
    fn synchronous_success_and_language_error_pass_through() {
        let source = CancellationSource::new();
        let token = source.token();
        assert_eq!(
            futures::executor::block_on(until_cancelled(&token, async { Ok::<_, &str>(3) })),
            Some(Ok(3))
        );
        assert_eq!(
            futures::executor::block_on(until_cancelled(&token, async {
                Err::<i32, _>("language error")
            })),
            Some(Err("language error"))
        );
    }

    #[test]
    fn cancellation_during_first_pending_poll_cannot_be_lost() {
        let source = CancellationSource::new();
        let token = source.token();
        let polls = Cell::new(0);
        let operation = poll_fn(|_| {
            polls.set(polls.get() + 1);
            source.cancel(CancellationReason::User);
            Poll::<()>::Pending
        });
        assert_eq!(
            futures::executor::block_on(until_cancelled(&token, operation)),
            None
        );
        assert_eq!(polls.get(), 1);

        let source = CancellationSource::new();
        let token = source.token();
        assert_eq!(
            futures::executor::block_on(until_cancelled(&token, async {
                source.cancel(CancellationReason::User);
                Err::<(), _>("provider observed cancellation")
            })),
            None,
            "a same-poll provider error cannot mask invocation cancellation"
        );
    }
}
