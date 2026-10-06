//! One selected-provider writer authority shared by cell and spreadsheet sinks.

use std::collections::HashMap;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use super::cooperation::TabularCooperation;
use super::path::TabularFsPath;
use crate::context::RuntimeContext;
use crate::execution::staged_file;
use futures::lock::Mutex as AsyncMutex;
use runmat_filesystem::FsStagedOutput;

#[path = "write_owner/publication.rs"]
mod publication;
pub(super) use publication::error_disposition as publication_error_disposition;
pub(super) use publication::interruption_error as publication_interruption_error;

pub(super) type WriteLock = Arc<AsyncMutex<()>>;
type WeakWriteLock = Weak<AsyncMutex<()>>;
type LockKey = (usize, PathBuf);
static WRITE_LOCKS: OnceLock<Mutex<HashMap<LockKey, WeakWriteLock>>> = OnceLock::new();

pub(super) async fn write_lock_for_path(selected: &TabularFsPath) -> io::Result<WriteLock> {
    let key = (
        selected.provider_identity(),
        write_lock_key(selected).await?,
    );
    let locks = WRITE_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut locks = locks.lock().expect("tabular write lock registry poisoned");
    if let Some(lock) = locks.get(&key).and_then(Weak::upgrade) {
        return Ok(lock);
    }
    locks.retain(|_, lock| lock.strong_count() > 0);
    let lock = Arc::new(AsyncMutex::new(()));
    locks.insert(key, Arc::downgrade(&lock));
    Ok(lock)
}

async fn write_lock_key(selected: &TabularFsPath) -> io::Result<PathBuf> {
    let path = selected.path();
    if let Ok(canonical) = selected.canonicalize_on_owner(path).await {
        return Ok(canonical);
    }
    selected.check()?;
    let mut candidate = path;
    let mut suffix = PathBuf::new();
    loop {
        if let Ok(canonical) = selected.canonicalize_on_owner(candidate).await {
            return Ok(if suffix.as_os_str().is_empty() {
                canonical
            } else {
                canonical.join(suffix)
            });
        }
        selected.check()?;
        let Some(name) = candidate.file_name() else {
            break;
        };
        let mut next_suffix = PathBuf::from(name);
        if !suffix.as_os_str().is_empty() {
            next_suffix.push(&suffix);
        }
        suffix = next_suffix;
        let Some(parent) = candidate.parent() else {
            break;
        };
        if parent == candidate {
            break;
        }
        candidate = parent;
    }
    selected.check()?;
    Ok(lexical_normalize(path))
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    normalized
}

pub(super) use staged_file::StagedFileError as StagedWriteError;

pub(super) async fn safe_replace_file(
    runtime: Option<&RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    builtin: &'static str,
    selected: &TabularFsPath,
    bytes: &[u8],
) -> Result<(), StagedWriteError> {
    let mut stage = begin_stage(cooperation, selected).await?;
    write_stage_bytes(
        runtime,
        cooperation,
        builtin,
        selected,
        stage.as_mut(),
        bytes,
    )
    .await?;
    finish_stage(runtime, cooperation, builtin, selected, stage).await
}

pub(super) async fn begin_stage(
    cooperation: &mut TabularCooperation<'_>,
    selected: &TabularFsPath,
) -> Result<Box<dyn FsStagedOutput>, StagedWriteError> {
    staged_file::begin_stage(
        cooperation,
        || selected.check(),
        || selected.begin_staged_replace(),
        "selected filesystem provider cannot stage a tabular replacement",
    )
    .await
}

pub(super) async fn write_stage_bytes(
    runtime: Option<&RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    builtin: &'static str,
    selected: &TabularFsPath,
    stage: &mut dyn FsStagedOutput,
    bytes: &[u8],
) -> Result<(), StagedWriteError> {
    staged_file::write_stage_bytes(
        runtime,
        cooperation,
        builtin,
        || selected.check(),
        stage,
        bytes,
        staged_file::StagedWriteMessages {
            no_progress: "staged tabular write made no progress",
            excess_progress: "staged tabular write reported excess progress",
        },
    )
    .await
}

pub(super) async fn finish_stage(
    runtime: Option<&RuntimeContext>,
    cooperation: &mut TabularCooperation<'_>,
    builtin: &'static str,
    selected: &TabularFsPath,
    stage: Box<dyn FsStagedOutput>,
) -> Result<(), StagedWriteError> {
    staged_file::finish_stage(runtime, cooperation, builtin, || selected.check(), stage).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::io::tabular::path::TabularFsOwner;
    use crate::execution::staged_file::PublicationInterruption;
    use crate::execution::RuntimeExecutionService;
    use async_trait::async_trait;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_filesystem::{self as fs, MemoryFsProvider};
    use runmat_filesystem::{FsPublicationDisposition, FsPublicationFailure};
    use std::error::Error;
    use std::future::Future;
    use std::rc::Rc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::{Context, Poll};

    struct PendingPublication {
        polled: Arc<AtomicBool>,
        dropped: Arc<AtomicBool>,
    }

    impl Drop for PendingPublication {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::SeqCst);
        }
    }

    #[async_trait(?Send)]
    impl FsStagedOutput for PendingPublication {
        async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }

        async fn flush_and_sync(&mut self) -> io::Result<()> {
            Ok(())
        }

        async fn publish(self: Box<Self>) -> Result<(), FsPublicationFailure> {
            self.polled.store(true, Ordering::SeqCst);
            futures::future::pending::<()>().await;
            unreachable!("pending publication fixture")
        }
    }

    struct ReadyCancellingPublication(CancellationSource);

    #[async_trait(?Send)]
    impl FsStagedOutput for ReadyCancellingPublication {
        async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }

        async fn flush_and_sync(&mut self) -> io::Result<()> {
            Ok(())
        }

        async fn publish(self: Box<Self>) -> Result<(), FsPublicationFailure> {
            self.0.cancel(runmat_execution::CancellationReason::User);
            Ok(())
        }
    }

    fn root() -> &'static str {
        "/tabular"
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replacement_and_lock_remain_on_selected_provider_after_ambient_swap() {
        let _lock = fs::provider_override_lock();
        let destination = PathBuf::from(root()).join("report.csv");
        let original = Arc::new(MemoryFsProvider::with_current_dir(root()));
        original
            .write_project_path(&destination, b"old\n")
            .expect("original fixture");
        let _original_guard = fs::replace_provider(original.clone());
        fs::set_current_dir(root()).expect("original cwd");
        let selected = TabularFsOwner::capture(Some(&destination))
            .expect("capture")
            .path(destination.clone())
            .expect("selected path");
        let original_lock =
            futures::executor::block_on(write_lock_for_path(&selected)).expect("original lock");

        let replacement = Arc::new(MemoryFsProvider::with_current_dir(root()));
        replacement
            .write_project_path(&destination, b"other\n")
            .expect("replacement fixture");
        let _replacement_guard = fs::replace_provider(replacement.clone());
        fs::set_current_dir(root()).expect("replacement cwd");
        let replacement_selected = TabularFsOwner::capture(Some(&destination))
            .expect("capture replacement")
            .path(destination.clone())
            .expect("replacement path");
        let replacement_lock =
            futures::executor::block_on(write_lock_for_path(&replacement_selected))
                .expect("replacement lock");
        assert!(!Arc::ptr_eq(&original_lock, &replacement_lock));

        let mut cooperation = TabularCooperation::new(None, "writecell");
        futures::executor::block_on(safe_replace_file(
            None,
            &mut cooperation,
            "writecell",
            &selected,
            b"new\n",
        ))
        .expect("replace with selected owner");
        assert_eq!(
            original.read_project_path(&destination).expect("original"),
            b"new\n"
        );
        assert_eq!(
            replacement
                .read_project_path(&destination)
                .expect("replacement"),
            b"other\n"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cancelled_pending_publication_is_indeterminate_and_drops_owner() {
        let _lock = fs::provider_override_lock();
        let provider = Arc::new(MemoryFsProvider::with_current_dir(root()));
        let destination = PathBuf::from(root()).join("pending.csv");
        provider
            .write_project_path(&destination, b"old")
            .expect("original target");
        let _restore = fs::replace_provider(provider.clone());
        fs::set_current_dir(root()).expect("selected cwd");
        let selected = TabularFsOwner::capture(Some(&destination))
            .expect("capture")
            .path(destination.clone())
            .expect("selected path");
        let source = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            source.token(),
        );
        let mut cooperation = TabularCooperation::new(Some(&runtime), "writecell");
        let polled = Arc::new(AtomicBool::new(false));
        let dropped = Arc::new(AtomicBool::new(false));
        let stage: Box<dyn FsStagedOutput> = Box::new(PendingPublication {
            polled: polled.clone(),
            dropped: dropped.clone(),
        });
        let mut call = Box::pin(finish_stage(
            Some(&runtime),
            &mut cooperation,
            "writecell",
            &selected,
            stage,
        ));
        let waker = futures::task::noop_waker();
        assert!(matches!(
            call.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        assert!(polled.load(Ordering::SeqCst));
        source.cancel(runmat_execution::CancellationReason::User);
        let StagedWriteError::PublicationInterrupted(cancelled, disposition) =
            futures::executor::block_on(call).expect_err("publication cancelled")
        else {
            panic!("pending publication must retain cancellation and uncertainty")
        };
        assert_eq!(disposition, FsPublicationDisposition::Indeterminate);
        let error = publication_interruption_error(cancelled, disposition);
        assert!(error
            .identifier()
            .is_some_and(|id| id.ends_with(":Cancelled")));
        let interruption = error
            .source()
            .and_then(|source| source.downcast_ref::<PublicationInterruption>())
            .expect("typed publication interruption");
        assert_eq!(
            interruption.disposition(),
            FsPublicationDisposition::Indeterminate
        );
        assert!(matches!(
                interruption.source().and_then(|source| source
                    .downcast_ref::<crate::execution::ExecutionServiceError>(
                )),
                Some(crate::execution::ExecutionServiceError::Cancelled)
            ));
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(provider.read_project_path(&destination).unwrap(), b"old");
    }

    #[test]
    fn committed_publication_failure_retains_disposition_and_source() {
        let failure = FsPublicationFailure::new(
            FsPublicationDisposition::Published,
            io::Error::new(io::ErrorKind::PermissionDenied, "postcommit lease revoked"),
        );
        assert_eq!(failure.disposition(), FsPublicationDisposition::Published);
        assert_eq!(failure.io_error().kind(), io::ErrorKind::PermissionDenied);
        assert!(failure.source().is_some());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cancellation_after_ready_publication_reports_published() {
        let _lock = fs::provider_override_lock();
        let provider = Arc::new(MemoryFsProvider::with_current_dir(root()));
        let _restore = fs::replace_provider(provider);
        fs::set_current_dir(root()).expect("selected cwd");
        let destination = PathBuf::from(root()).join("ready.csv");
        let selected = TabularFsOwner::capture(Some(&destination))
            .expect("capture")
            .path(destination)
            .expect("selected path");
        let source = CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            source.token(),
        );
        let mut cooperation = TabularCooperation::new(Some(&runtime), "writecell");
        let result = futures::executor::block_on(finish_stage(
            Some(&runtime),
            &mut cooperation,
            "writecell",
            &selected,
            Box::new(ReadyCancellingPublication(source)),
        ));
        let StagedWriteError::PublicationInterrupted(_, disposition) =
            result.expect_err("post-publication cancellation")
        else {
            panic!("ready publication must retain its committed disposition")
        };
        assert_eq!(disposition, FsPublicationDisposition::Published);
    }
}
