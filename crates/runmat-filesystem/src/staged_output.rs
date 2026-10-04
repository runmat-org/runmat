//! Provider-owned staging for one file replacement.
//!
//! A staged output is not a filename handed to its consumer. The provider
//! keeps the exclusive staging token and is responsible for aborting it when
//! any owning future is dropped. This is separate from resource certification.

use async_trait::async_trait;
use std::io;

/// An unpublished output owned by the filesystem provider that created it.
///
/// `publish` may suspend for a future provider implementation, but it must
/// have one atomic, provider-defined commit point. Before that point, dropping
/// the future aborts the stage without modifying the destination. After that
/// point, the future must finish in the same poll without another suspension:
/// a caller cannot observe cancellation while publication continues unseen.
/// Implementations must not detach a commit task from this owner.
#[async_trait(?Send)]
pub trait FsStagedOutput: Send {
    /// Write at most `bytes.len()` bytes; zero is permitted only for empty input.
    /// Callers own partial-write, Interrupted, cancellation, and lease checks.
    async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize>;

    /// Make the staged bytes ready for publication without changing the target.
    async fn flush_and_sync(&mut self) -> io::Result<()>;

    /// Publish this exact stage over its captured destination. A failed or
    /// dropped publication retains the previous target and aborts the stage.
    async fn publish(self: Box<Self>) -> io::Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::future::pending;
    use futures::task::noop_waker;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::task::{Context, Poll};

    struct PendingOutput {
        dropped: Arc<AtomicBool>,
    }

    impl Drop for PendingOutput {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait(?Send)]
    impl FsStagedOutput for PendingOutput {
        async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len().min(2))
        }

        async fn flush_and_sync(&mut self) -> io::Result<()> {
            Ok(())
        }

        async fn publish(self: Box<Self>) -> io::Result<()> {
            pending::<()>().await;
            unreachable!("fixture never publishes")
        }
    }

    #[test]
    fn partial_progress_and_dropped_pending_publication_retain_owner() {
        let dropped = Arc::new(AtomicBool::new(false));
        let mut stage: Box<dyn FsStagedOutput> = Box::new(PendingOutput {
            dropped: dropped.clone(),
        });
        assert_eq!(
            futures::executor::block_on(stage.write_chunk(b"abcd")).unwrap(),
            2
        );

        let mut future = Box::pin(stage.publish());
        let waker = noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(
            Future::poll(Pin::as_mut(&mut future), &mut context),
            Poll::Pending
        ));
        assert!(!dropped.load(Ordering::SeqCst));
        drop(future);
        assert!(dropped.load(Ordering::SeqCst));
    }
}
