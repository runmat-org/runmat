//! Captured filesystem authority for one PNG export.

use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_filesystem::{self as fs, CapturedFsResourceLease, FsProvider, FsPublicationFailure};

use super::{print_error_with_source, BUILTIN_NAME, PRINT_ERROR_IO};
use crate::context::RuntimeContext;
use crate::execution::staged_file::{self, StagedCpuCooperation, StagedFileError};
use crate::{BuiltinResult, RuntimeError};

pub(super) struct PrintFsOwner {
    provider: Arc<dyn FsProvider>,
    cwd: Option<PathBuf>,
    lease: Option<CapturedFsResourceLease>,
}

pub(super) struct PrintFsPath {
    provider: Arc<dyn FsProvider>,
    path: PathBuf,
    lease: Option<CapturedFsResourceLease>,
}

impl PrintFsOwner {
    /// The filename may be resident, so retain the selected provider and CWD
    /// before the first materialization await.
    pub(super) fn capture(known_path: Option<&Path>) -> io::Result<Self> {
        let (provider, cwd) = if let Some(path) = known_path {
            fs::capture_provider_for_paths(&[path])?
        } else {
            let (provider, cwd) = fs::current_provider_with_current_dir()?;
            (provider, Some(cwd))
        };
        let lease = fs::capture_resource_lease_for_effect(provider.clone())?;
        let owner = Self {
            provider,
            cwd,
            lease,
        };
        owner.check()?;
        Ok(owner)
    }

    pub(super) fn path(self, path: PathBuf) -> io::Result<PrintFsPath> {
        let path = if self.provider.path_namespace().is_absolute(&path) {
            path
        } else {
            self.cwd
                .as_ref()
                .ok_or_else(|| io::Error::other("relative PNG path has no captured directory"))?
                .join(path)
        };
        let selected = PrintFsPath {
            provider: self.provider,
            path,
            lease: self.lease,
        };
        selected.check()?;
        Ok(selected)
    }

    fn check(&self) -> io::Result<()> {
        check(self.lease.as_ref())
    }
}

impl PrintFsPath {
    fn check(&self) -> io::Result<()> {
        check(self.lease.as_ref())
    }

    #[cfg(test)]
    pub(super) fn path_name(&self) -> &Path {
        &self.path
    }
}

fn check(lease: Option<&CapturedFsResourceLease>) -> io::Result<()> {
    if lease.is_some_and(|lease| !lease.is_valid()) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "print filesystem authority was revoked",
        ));
    }
    Ok(())
}

pub(super) fn map_io(source: io::Error) -> RuntimeError {
    print_error_with_source(
        format!("{}: {source}", PRINT_ERROR_IO.message),
        &PRINT_ERROR_IO,
        source,
    )
}

pub(super) async fn await_effect<T>(
    runtime: Option<&RuntimeContext>,
    work: impl Future<Output = T>,
) -> BuiltinResult<T> {
    staged_file::await_cancellable(runtime, BUILTIN_NAME, work).await
}

pub(super) async fn replace_png(
    runtime: Option<&RuntimeContext>,
    selected: &PrintFsPath,
    payload: &[u8],
) -> BuiltinResult<()> {
    let mut cooperation = StagedCpuCooperation::new(runtime, BUILTIN_NAME);
    let mut stage = staged_file::begin_stage(
        &mut cooperation,
        || selected.check(),
        || selected.provider.begin_staged_replace(&selected.path),
        "selected filesystem provider cannot stage PNG replacement",
    )
    .await
    .map_err(map_staged_error)?;
    staged_file::write_stage_bytes(
        runtime,
        &mut cooperation,
        BUILTIN_NAME,
        || selected.check(),
        stage.as_mut(),
        payload,
        staged_file::StagedWriteMessages {
            no_progress: "staged PNG write made no progress",
            excess_progress: "staged PNG write reported excess progress",
        },
    )
    .await
    .map_err(map_staged_error)?;
    staged_file::finish_stage(
        runtime,
        &mut cooperation,
        BUILTIN_NAME,
        || selected.check(),
        stage,
    )
    .await
    .map_err(map_staged_error)
}

fn map_staged_error(error: StagedFileError) -> RuntimeError {
    match error {
        StagedFileError::Io(source) => map_io(source),
        StagedFileError::Control(error) => error,
        StagedFileError::Publication(source) => map_publication(source),
        StagedFileError::PublicationInterrupted(error, disposition) => {
            staged_file::interruption_error(error, disposition)
        }
    }
}

fn map_publication(source: FsPublicationFailure) -> RuntimeError {
    print_error_with_source(
        format!("{}: {source}", PRINT_ERROR_IO.message),
        &PRINT_ERROR_IO,
        source,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::executor::block_on;
    use futures::task::noop_waker;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_filesystem::FsPublicationDisposition;
    use runmat_filesystem::MemoryFsProvider;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll};

    struct PendingStage {
        polls: Arc<AtomicUsize>,
        dropped: Arc<AtomicBool>,
    }

    struct TerminalStage {
        cancel: CancellationSource,
        failure: bool,
    }

    #[async_trait(?Send)]
    impl fs::FsStagedOutput for TerminalStage {
        async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }
        async fn flush_and_sync(&mut self) -> io::Result<()> {
            Ok(())
        }
        async fn publish(self: Box<Self>) -> Result<(), FsPublicationFailure> {
            self.cancel
                .cancel(runmat_execution::CancellationReason::User);
            if self.failure {
                Err(FsPublicationFailure::not_published(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "publication rejected",
                )))
            } else {
                Ok(())
            }
        }
    }

    impl Drop for PendingStage {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait(?Send)]
    impl fs::FsStagedOutput for PendingStage {
        async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }
        async fn flush_and_sync(&mut self) -> io::Result<()> {
            Ok(())
        }
        async fn publish(self: Box<Self>) -> Result<(), FsPublicationFailure> {
            self.polls.fetch_add(1, Ordering::SeqCst);
            futures::future::pending::<()>().await;
            unreachable!("pending stage never publishes")
        }
    }

    #[test]
    fn selected_provider_and_cwd_survive_ambient_swap_before_staged_publish() {
        let _lock = fs::provider_override_lock();
        let selected = Arc::new(MemoryFsProvider::with_current_dir("/selected"));
        let replacement = Arc::new(MemoryFsProvider::with_current_dir("/replacement"));
        let _restore = fs::replace_provider(selected.clone());
        fs::set_current_dir("/selected").expect("selected directory");
        let owner = PrintFsOwner::capture(None).expect("capture selected owner");
        fs::set_provider(replacement.clone());
        fs::set_current_dir("/replacement").expect("replacement directory");
        let path = owner
            .path(PathBuf::from("plot.png"))
            .expect("selected path");
        assert_eq!(path.path_name(), Path::new("/selected/plot.png"));
        block_on(replace_png(None, &path, b"\x89PNG\r\n\x1a\nselected"))
            .expect("selected stage publishes");
        assert_eq!(
            selected.read_project_path("/selected/plot.png").unwrap(),
            b"\x89PNG\r\n\x1a\nselected"
        );
        assert!(replacement
            .read_project_path("/replacement/plot.png")
            .is_err());
    }

    #[test]
    fn absolute_host_filename_captures_provider_without_cwd_dependency() {
        let _lock = fs::provider_override_lock();
        let selected = Arc::new(MemoryFsProvider::with_current_dir("/selected"));
        let _restore = fs::replace_provider(selected);
        let owner =
            PrintFsOwner::capture(Some(Path::new("/exports/plot.png"))).expect("absolute capture");
        assert!(owner.cwd.is_none());
        let selected = owner
            .path(PathBuf::from("/exports/plot.png"))
            .expect("absolute path");
        assert_eq!(selected.path_name(), Path::new("/exports/plot.png"));
    }

    #[test]
    fn all_admitted_host_text_forms_capture_one_mounted_provider() {
        let _lock = fs::provider_override_lock();
        let selected = Arc::new(MemoryFsProvider::with_current_dir("/selected"));
        let replacement = Arc::new(MemoryFsProvider::with_current_dir("/replacement"));
        let _restore = fs::replace_provider(selected.clone());
        let name = "/exports/figure.png";
        selected
            .write_project_path(name, b"old")
            .expect("selected parent");
        let values = [
            runmat_value::Value::String(name.into()),
            runmat_value::Value::CharArray(runmat_value::CharArray::new_row(name)),
            runmat_value::Value::StringArray(
                runmat_value::StringArray::new_2d(vec![name.into()], 1, 1).unwrap(),
            ),
        ];
        let owners: Vec<_> = values
            .iter()
            .map(|value| {
                let path = super::super::preflight_host_arguments(
                    crate::call::FunctionArgumentView::from_slice(std::slice::from_ref(value)),
                )
                .expect("valid text")
                .expect("known host filename");
                PrintFsOwner::capture(Some(&path)).expect("capture mounted provider")
            })
            .collect();
        fs::set_provider(replacement.clone());
        for owner in owners {
            assert!(owner.cwd.is_none());
            let selected_path = owner.path(PathBuf::from(name)).expect("captured path");
            block_on(replace_png(None, &selected_path, b"png")).expect("publish on original mount");
        }
        assert_eq!(selected.read_project_path(name).unwrap(), b"png");
        assert!(replacement.read_project_path(name).is_err());
    }

    #[test]
    fn malformed_control_preflight_never_opens_a_stage() {
        let _lock = fs::provider_override_lock();
        let selected = Arc::new(MemoryFsProvider::with_current_dir("/selected"));
        let _restore = fs::replace_provider(selected.clone());
        fs::set_current_dir("/selected").expect("selected directory");
        assert!(super::super::preflight_host_arguments(
            crate::call::FunctionArgumentView::from_slice(&[
                runmat_value::Value::from("plot"),
                runmat_value::Value::from("-dsvg"),
            ])
        )
        .is_err());
        assert!(selected.read_project_path("/selected/plot.png").is_err());
    }

    #[test]
    fn dropped_pending_publication_retires_exact_stage_without_replay() {
        let polls = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        let stage = Box::new(PendingStage {
            polls: polls.clone(),
            dropped: dropped.clone(),
        });
        let mut call = Box::pin(staged_file::publish(None, BUILTIN_NAME, stage));
        let waker = noop_waker();
        assert!(matches!(
            call.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        drop(call);
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(polls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cancelled_pending_png_publication_preserves_indeterminate_disposition() {
        let source = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        let polls = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        let stage = Box::new(PendingStage {
            polls: polls.clone(),
            dropped: dropped.clone(),
        });
        let mut call = Box::pin(staged_file::publish(Some(&runtime), BUILTIN_NAME, stage));
        let waker = noop_waker();
        assert!(matches!(
            call.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        source.cancel(runmat_execution::CancellationReason::User);
        let error = block_on(call).expect_err("cancelled publication");
        let StagedFileError::PublicationInterrupted(error, disposition) = error else {
            panic!("cancelled publication retains its exact dispatch state");
        };
        assert_eq!(disposition, FsPublicationDisposition::Indeterminate);
        let diagnostic =
            map_staged_error(StagedFileError::PublicationInterrupted(error, disposition));
        assert!(diagnostic
            .identifier()
            .is_some_and(|id| id.ends_with(":Cancelled")));
        assert_eq!(
            staged_file::error_disposition(&diagnostic),
            Some(FsPublicationDisposition::Indeterminate)
        );
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(polls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn terminal_png_publication_outcome_wins_simultaneous_cancel() {
        for failure in [false, true] {
            let source = CancellationSource::new();
            let runtime = RuntimeContext::with_cancellation(
                Rc::new(crate::execution::RuntimeExecutionService::new()),
                source.token(),
            );
            let result = block_on(staged_file::publish(
                Some(&runtime),
                BUILTIN_NAME,
                Box::new(TerminalStage {
                    cancel: source,
                    failure,
                }),
            ));
            match (failure, result) {
                (
                    false,
                    Err(StagedFileError::PublicationInterrupted(
                        error,
                        FsPublicationDisposition::Published,
                    )),
                ) => {
                    assert!(error
                        .identifier()
                        .is_some_and(|id| id.ends_with(":Cancelled")));
                }
                (true, Err(StagedFileError::Publication(cause))) => {
                    assert_eq!(cause.disposition(), FsPublicationDisposition::NotPublished);
                    assert_eq!(cause.io_error().kind(), io::ErrorKind::PermissionDenied);
                }
                _ => panic!("provider terminal outcome must win cancellation"),
            }
        }
    }
}
