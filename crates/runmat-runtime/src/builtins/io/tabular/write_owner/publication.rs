//! One publication attempt retains its exact poll and commit disposition.

use std::cell::{Cell, RefCell};
use std::future::{poll_fn, Future};

use crate::builtins::io::tabular::cooperation::await_io;
use crate::context::RuntimeContext;
use crate::RuntimeError;
use runmat_filesystem::{FsPublicationDisposition, FsStagedOutput};

use super::StagedWriteError;

/// Keep the original runtime diagnostic, policy, context, and typed source.
pub(in crate::builtins::io::tabular) fn interruption_error(
    mut error: RuntimeError,
    disposition: FsPublicationDisposition,
) -> RuntimeError {
    let source = error.source.take();
    error.source = Some(Box::new(PublicationInterruption {
        disposition,
        source,
    }));
    error
}

#[derive(Debug)]
pub(super) struct PublicationInterruption {
    disposition: FsPublicationDisposition,
    source: Option<Box<runmat_async::runtime_error::RuntimeErrorSource>>,
}

impl PublicationInterruption {
    pub(super) fn disposition(&self) -> FsPublicationDisposition {
        self.disposition
    }
}

impl std::fmt::Display for PublicationInterruption {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "staged publication outcome is {:?}",
            self.disposition()
        )
    }
}

impl std::error::Error for PublicationInterruption {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.source
            .as_ref()
            .map(|source| source.as_ref() as &(dyn std::error::Error + 'static))
    }
}

pub(super) async fn publish(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    stage: Box<dyn FsStagedOutput>,
) -> Result<(), StagedWriteError> {
    let polled = Cell::new(false);
    let completed = RefCell::new(None);
    let mut publication = stage.publish();
    let result = await_io(
        runtime,
        builtin,
        poll_fn(|context| {
            polled.set(true);
            match publication.as_mut().poll(context) {
                std::task::Poll::Ready(outcome) => {
                    *completed.borrow_mut() = Some(outcome);
                    std::task::Poll::Ready(())
                }
                std::task::Poll::Pending => std::task::Poll::Pending,
            }
        }),
    )
    .await;
    match (result, completed.into_inner()) {
        (_, Some(Err(failure))) => Err(StagedWriteError::Publication(failure)),
        (Err(error), Some(Ok(()))) => Err(StagedWriteError::PublicationInterrupted(
            error,
            FsPublicationDisposition::Published,
        )),
        (Err(error), None) => Err(StagedWriteError::PublicationInterrupted(
            error,
            if polled.get() {
                FsPublicationDisposition::Indeterminate
            } else {
                FsPublicationDisposition::NotPublished
            },
        )),
        (Ok(()), Some(Ok(()))) => Ok(()),
        (Ok(()), None) => unreachable!("publication future completed without an outcome"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use async_trait::async_trait;
    use runmat_execution::cancellation::CancellationSource;
    use std::io;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    struct CancelWithFailure {
        source: CancellationSource,
        polled: Arc<AtomicBool>,
    }

    #[async_trait(?Send)]
    impl FsStagedOutput for CancelWithFailure {
        async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }

        async fn flush_and_sync(&mut self) -> io::Result<()> {
            Ok(())
        }

        async fn publish(self: Box<Self>) -> Result<(), runmat_filesystem::FsPublicationFailure> {
            self.polled.store(true, Ordering::SeqCst);
            self.source
                .cancel(runmat_execution::CancellationReason::User);
            Err(runmat_filesystem::FsPublicationFailure::not_published(
                io::Error::new(io::ErrorKind::PermissionDenied, "provider refused commit"),
            ))
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn provider_failure_cause_survives_simultaneous_cancellation() {
        let source = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            source.token(),
        );
        let polled = Arc::new(AtomicBool::new(false));
        let stage: Box<dyn FsStagedOutput> = Box::new(CancelWithFailure {
            source,
            polled: polled.clone(),
        });
        let result = futures::executor::block_on(publish(Some(&runtime), "writecell", stage));
        let StagedWriteError::Publication(failure) = result.expect_err("provider failure") else {
            panic!("provider failure must retain its source over cancellation")
        };
        assert!(polled.load(Ordering::SeqCst));
        assert_eq!(
            failure.disposition(),
            FsPublicationDisposition::NotPublished
        );
        assert_eq!(failure.io_error().kind(), io::ErrorKind::PermissionDenied);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn pre_cancelled_publication_does_not_poll_provider() {
        let source = CancellationSource::new();
        source.cancel(runmat_execution::CancellationReason::User);
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            source.token(),
        );
        let polled = Arc::new(AtomicBool::new(false));
        let stage: Box<dyn FsStagedOutput> = Box::new(CancelWithFailure {
            source,
            polled: polled.clone(),
        });
        let result = futures::executor::block_on(publish(Some(&runtime), "writecell", stage));
        let StagedWriteError::PublicationInterrupted(_, disposition) =
            result.expect_err("cancelled before provider publication")
        else {
            panic!("pre-cancelled publication must retain typed disposition")
        };
        assert_eq!(disposition, FsPublicationDisposition::NotPublished);
        assert!(!polled.load(Ordering::SeqCst));
    }
}
