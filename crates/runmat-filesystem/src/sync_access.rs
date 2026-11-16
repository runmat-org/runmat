//! Synchronous entry points over an asynchronous provider.
//!
//! In the browser a provider read usually resolves through a JavaScript promise. A promise
//! settles only after the worker returns to its event loop, so a thread that blocks on it waits
//! forever; without wasm atomics the executor's park returns at once and the wait becomes a
//! busy loop at full CPU. The browser's synchronous entry points therefore poll the operation
//! once: an operation the provider completes immediately (an in-memory provider, a JavaScript
//! function that returns its value directly) returns its result, and an operation that would
//! wait fails at once with [`SynchronousAccessUnavailable`]. Callers in async code use the
//! `*_async` functions instead.

use std::fmt;
use std::future::Future;
use std::io::{self, ErrorKind};
use std::pin::pin;
use std::task::{Context, Poll};

/// A synchronous filesystem call reached an operation that can only finish asynchronously.
///
/// It travels inside an [`io::Error`] of kind [`ErrorKind::WouldBlock`]; use
/// [`is_synchronous_access_unavailable`] to recognize it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynchronousAccessUnavailable {
    pub operation: &'static str,
}

impl fmt::Display for SynchronousAccessUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "synchronous file access is not available in the browser: `{}` must read \
asynchronously",
            self.operation
        )
    }
}

impl std::error::Error for SynchronousAccessUnavailable {}

/// Whether `error` is a [`SynchronousAccessUnavailable`] failure.
pub fn is_synchronous_access_unavailable(error: &io::Error) -> bool {
    error.kind() == ErrorKind::WouldBlock
        && error
            .get_ref()
            .is_some_and(|inner| inner.is::<SynchronousAccessUnavailable>())
}

/// Polls `future` once. A ready result is returned; a pending one fails with
/// [`SynchronousAccessUnavailable`] for `operation` and the future is dropped.
pub(crate) fn poll_once<T>(
    operation: &'static str,
    future: impl Future<Output = io::Result<T>>,
) -> io::Result<T> {
    let mut future = pin!(future);
    let mut context = Context::from_waker(futures::task::noop_waker_ref());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(result) => result,
        Poll::Pending => Err(io::Error::new(
            ErrorKind::WouldBlock,
            SynchronousAccessUnavailable { operation },
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn a_pending_operation_fails_at_once_with_the_typed_error() {
        let started = Instant::now();
        let error = poll_once("read", futures::future::pending::<io::Result<Vec<u8>>>())
            .expect_err("a pending read cannot complete synchronously");
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(is_synchronous_access_unavailable(&error), "{error}");
        assert!(error
            .to_string()
            .contains("`read` must read asynchronously"));
    }

    #[test]
    fn an_operation_that_completes_immediately_returns_its_result() {
        let bytes = poll_once("read", async { Ok(b"part".to_vec()) }).unwrap();
        assert_eq!(bytes, b"part");
        let error = poll_once::<()>("read", async { Err(io::Error::from(ErrorKind::NotFound)) })
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::NotFound);
        assert!(!is_synchronous_access_unavailable(&error));
    }

    /// The wasm `wait_for_fs` is `poll_once` over the provider future; a provider whose reads
    /// settle later (a JavaScript promise) reaches the same pending branch.
    #[test]
    fn a_provider_read_that_settles_later_does_not_block() {
        struct Later(bool);
        impl Future for Later {
            type Output = io::Result<Vec<u8>>;
            fn poll(
                mut self: std::pin::Pin<&mut Self>,
                cx: &mut Context<'_>,
            ) -> Poll<Self::Output> {
                if self.0 {
                    return Poll::Ready(Ok(Vec::new()));
                }
                self.0 = true;
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
        let error = poll_once("read_dir", Later(false)).unwrap_err();
        assert!(is_synchronous_access_unavailable(&error));
    }
}
