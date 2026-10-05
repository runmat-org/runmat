//! Provider-owned staging for one file replacement.
//!
//! A staged output is not a filename handed to its consumer. The provider
//! keeps the exclusive staging token and is responsible for aborting it when
//! any owning future is dropped. This is separate from resource certification.

use async_trait::async_trait;
use std::io;
use std::{error::Error, fmt};

/// What is known about the destination after a publication failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FsPublicationDisposition {
    /// The previous destination was not changed.
    NotPublished,
    /// The new bytes were committed, but a later step failed.
    Published,
    /// The issuer cannot prove which version is public; do not retry blindly.
    Indeterminate,
}

/// Preserve both the provider failure and its observable commit disposition.
#[derive(Debug)]
pub struct FsPublicationFailure {
    disposition: FsPublicationDisposition,
    source: io::Error,
}

impl FsPublicationFailure {
    pub fn new(disposition: FsPublicationDisposition, source: io::Error) -> Self {
        Self {
            disposition,
            source,
        }
    }

    pub fn not_published(source: io::Error) -> Self {
        Self::new(FsPublicationDisposition::NotPublished, source)
    }

    pub fn disposition(&self) -> FsPublicationDisposition {
        self.disposition
    }

    pub fn io_error(&self) -> &io::Error {
        &self.source
    }
}

impl fmt::Display for FsPublicationFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?} staged publication: {}",
            self.disposition, self.source
        )
    }
}

impl Error for FsPublicationFailure {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

/// An unpublished output owned by the filesystem provider that created it.
///
/// `publish` has one provider-defined commit point. A provider must report
/// whether failure left the old destination intact, occurred after commit, or
/// cannot be determined. Dropping a stage before commit dispatch aborts its
/// private bytes. Dropping an in-flight remote publication may leave a server
/// commit underway; it must never be treated as proof of rollback. Providers
/// must not detach a local commit task from this owner, and distributed
/// transports need an explicit status/commit protocol before they can opt in.
#[async_trait(?Send)]
pub trait FsStagedOutput: Send {
    /// Write at most `bytes.len()` bytes; zero is permitted only for empty input.
    /// Callers own partial-write, Interrupted, cancellation, and lease checks.
    async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize>;

    /// Make the staged bytes ready for publication without changing the target.
    async fn flush_and_sync(&mut self) -> io::Result<()>;

    /// Publish this exact stage over its captured destination. An error must
    /// carry the actual commit disposition; only `NotPublished` promises the
    /// previous target is unchanged.
    async fn publish(self: Box<Self>) -> Result<(), FsPublicationFailure>;
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

        async fn publish(self: Box<Self>) -> Result<(), FsPublicationFailure> {
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
