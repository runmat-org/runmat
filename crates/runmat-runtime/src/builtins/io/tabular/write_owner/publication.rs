//! One publication attempt retains its exact poll and commit disposition.

#[cfg(test)]
use super::StagedWriteError;
#[cfg(test)]
use crate::context::RuntimeContext;
use crate::execution::staged_file;
#[cfg(test)]
use runmat_filesystem::FsStagedOutput;

pub(in crate::builtins::io::tabular) use staged_file::error_disposition;
pub(in crate::builtins::io::tabular) use staged_file::interruption_error;

#[cfg(test)]
pub(super) async fn publish(
    runtime: Option<&RuntimeContext>,
    builtin: &'static str,
    stage: Box<dyn FsStagedOutput>,
) -> Result<(), StagedWriteError> {
    staged_file::publish(runtime, builtin, stage).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use async_trait::async_trait;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_filesystem::FsPublicationDisposition;
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
