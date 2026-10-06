//! Captured filesystem authority for one PNG export.

use std::cell::{Cell, RefCell};
use std::fmt;
use std::future::{poll_fn, Future};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use runmat_filesystem::{
    self as fs, CapturedFsResourceLease, FsProvider, FsPublicationDisposition, FsPublicationFailure,
};

use super::{print_error_with_source, BUILTIN_NAME, PRINT_ERROR_IO};
use crate::context::RuntimeContext;
use crate::execution::{CpuCooperationBudget, ExecutionServiceError};
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

async fn checkpoint(
    runtime: Option<&RuntimeContext>,
    budget: &mut Option<CpuCooperationBudget>,
    bytes: usize,
) -> BuiltinResult<()> {
    let (Some(runtime), Some(budget)) = (runtime, budget.as_mut()) else {
        return Ok(());
    };
    budget
        .checkpoint(u64::try_from(bytes).unwrap_or(u64::MAX))
        .await
        .map_err(|source| execution_error(runtime, source))
}

async fn await_io<T>(
    runtime: Option<&RuntimeContext>,
    work: impl Future<Output = T>,
) -> BuiltinResult<T> {
    let Some(runtime) = runtime else {
        return Ok(work.await);
    };
    runtime
        .cpu_cooperation()
        .await_cancellable(work)
        .await
        .map_err(|source| execution_error(runtime, source))
}

pub(super) async fn await_effect<T>(
    runtime: Option<&RuntimeContext>,
    work: impl Future<Output = T>,
) -> BuiltinResult<T> {
    await_io(runtime, work).await
}

fn execution_error(runtime: &RuntimeContext, source: ExecutionServiceError) -> RuntimeError {
    match source {
        cancelled @ ExecutionServiceError::Cancelled => {
            crate::build_runtime_error("print: execution was cancelled")
                .with_builtin(BUILTIN_NAME)
                .with_identifier(format!("{}:Cancelled", runtime.error_namespace()))
                .with_source(cancelled)
                .build()
        }
        other => other.into_runtime_error(),
    }
}

pub(super) async fn replace_png(
    runtime: Option<&RuntimeContext>,
    selected: &PrintFsPath,
    payload: &[u8],
) -> BuiltinResult<()> {
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    checkpoint(runtime, &mut budget, 0).await?;
    selected.check().map_err(map_io)?;
    let mut stage = selected
        .provider
        .begin_staged_replace(&selected.path)
        .map_err(map_io)?
        .ok_or_else(|| {
            map_io(io::Error::new(
                io::ErrorKind::Unsupported,
                "selected filesystem provider cannot stage PNG replacement",
            ))
        })?;
    selected.check().map_err(map_io)?;

    for chunk in payload.chunks(4096) {
        let mut remaining = chunk;
        while !remaining.is_empty() {
            checkpoint(runtime, &mut budget, remaining.len()).await?;
            selected.check().map_err(map_io)?;
            let written = await_io(runtime, stage.write_chunk(remaining)).await?;
            match written {
                Ok(0) => {
                    return Err(map_io(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "staged PNG write made no progress",
                    )))
                }
                Ok(n) if n > remaining.len() => {
                    return Err(map_io(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "staged PNG write reported excess progress",
                    )))
                }
                Ok(n) => remaining = &remaining[n..],
                Err(source) if source.kind() == io::ErrorKind::Interrupted => continue,
                Err(source) => return Err(map_io(source)),
            }
        }
    }
    checkpoint(runtime, &mut budget, 0).await?;
    selected.check().map_err(map_io)?;
    await_io(runtime, stage.flush_and_sync())
        .await?
        .map_err(map_io)?;
    checkpoint(runtime, &mut budget, 0).await?;
    selected.check().map_err(map_io)?;
    publish(runtime, stage).await?;
    selected.check().map_err(|source| {
        map_publication(FsPublicationFailure::new(
            FsPublicationDisposition::Published,
            source,
        ))
    })
}

fn map_publication(source: FsPublicationFailure) -> RuntimeError {
    print_error_with_source(
        format!("{}: {source}", PRINT_ERROR_IO.message),
        &PRINT_ERROR_IO,
        source,
    )
}

/// A cancellation after commit dispatch is indeterminate unless the provider
/// returned an exact terminal disposition. Never report it as a rollback.
async fn publish(
    runtime: Option<&RuntimeContext>,
    stage: Box<dyn fs::FsStagedOutput>,
) -> BuiltinResult<()> {
    let polled = Cell::new(false);
    let completed = RefCell::new(None);
    let mut publication = stage.publish();
    let raced = await_io(
        runtime,
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
    match (raced, completed.into_inner()) {
        (_, Some(Err(failure))) => Err(map_publication(failure)),
        (Err(control), Some(Ok(()))) => Err(publication_interruption(
            control,
            FsPublicationDisposition::Published,
        )),
        (Err(control), None) => Err(publication_interruption(
            control,
            if polled.get() {
                FsPublicationDisposition::Indeterminate
            } else {
                FsPublicationDisposition::NotPublished
            },
        )),
        (Ok(()), Some(Ok(()))) => Ok(()),
        (Ok(()), None) => unreachable!("publication completed without provider outcome"),
    }
}

fn publication_interruption(
    mut control: RuntimeError,
    disposition: FsPublicationDisposition,
) -> RuntimeError {
    let source = control.source.take();
    control.source = Some(Box::new(PublicationInterruption {
        disposition,
        source,
    }));
    control
}

#[derive(Debug)]
struct PublicationInterruption {
    disposition: FsPublicationDisposition,
    source: Option<Box<runmat_async::runtime_error::RuntimeErrorSource>>,
}

impl fmt::Display for PublicationInterruption {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "PNG publication outcome is {:?}",
            self.disposition
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

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::executor::block_on;
    use futures::task::noop_waker;
    use runmat_filesystem::MemoryFsProvider;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll};

    struct PendingStage {
        polls: Arc<AtomicUsize>,
        dropped: Arc<AtomicBool>,
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
    fn malformed_control_preflight_never_opens_a_stage() {
        let _lock = fs::provider_override_lock();
        let selected = Arc::new(MemoryFsProvider::with_current_dir("/selected"));
        let _restore = fs::replace_provider(selected.clone());
        fs::set_current_dir("/selected").expect("selected directory");
        assert!(super::super::preflight_host_arguments(&[
            runmat_value::Value::from("plot"),
            runmat_value::Value::from("-dsvg"),
        ])
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
        let mut call = Box::pin(publish(None, stage));
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
}
