use super::*;
use async_trait::async_trait;
use futures::task::noop_waker_ref;
use runmat_filesystem::{
    DirEntry, FileHandle, FsMetadata, FsProvider, MemoryFsProvider, OpenFlags,
};
use std::future::Future;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TextOperation {
    Read,
    Write,
}

impl TextOperation {
    fn source(self) -> &'static str {
        match self {
            Self::Read => "readlines('/textlines/lines.txt');",
            Self::Write => "writelines('alpha', '/textlines/lines.txt');",
        }
    }
}

#[derive(Default)]
struct OperationEvidence {
    reads: AtomicUsize,
    writes: AtomicUsize,
    retired: AtomicUsize,
    completed: AtomicUsize,
}

impl OperationEvidence {
    fn calls(&self, operation: TextOperation) -> usize {
        match operation {
            TextOperation::Read => self.reads.load(Ordering::SeqCst),
            TextOperation::Write => self.writes.load(Ordering::SeqCst),
        }
    }
}

struct PendingOperation(Arc<OperationEvidence>);

impl Drop for PendingOperation {
    fn drop(&mut self) {
        self.0.retired.fetch_add(1, Ordering::SeqCst);
    }
}

struct TextProvider {
    memory: MemoryFsProvider,
    pending: Option<TextOperation>,
    evidence: Arc<OperationEvidence>,
}

impl TextProvider {
    fn new(pending: Option<TextOperation>) -> Self {
        Self {
            memory: MemoryFsProvider::with_current_dir("/textlines"),
            pending,
            evidence: Arc::new(OperationEvidence::default()),
        }
    }

    async fn suspend(&self) -> io::Result<()> {
        let _retirement = PendingOperation(self.evidence.clone());
        std::future::pending::<()>().await;
        self.evidence.completed.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[async_trait(?Send)]
impl FsProvider for TextProvider {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        self.memory.path_namespace()
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/textlines"))
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.memory.open(path, flags)
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.evidence.reads.fetch_add(1, Ordering::SeqCst);
        if self.pending == Some(TextOperation::Read) {
            self.suspend().await?;
        }
        self.memory.read(path).await
    }

    async fn write(&self, path: &Path, bytes: &[u8]) -> io::Result<()> {
        self.evidence.writes.fetch_add(1, Ordering::SeqCst);
        if self.pending == Some(TextOperation::Write) {
            self.suspend().await?;
        }
        self.memory.write(path, bytes).await
    }

    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_file(path).await
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.memory.metadata(path).await
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.memory.symlink_metadata(path).await
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.memory.read_dir(path).await
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.memory.canonicalize(path).await
    }

    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.create_dir(path).await
    }

    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.memory.create_dir_all(path).await
    }

    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir(path).await
    }

    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.memory.remove_dir_all(path).await
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.memory.rename(from, to).await
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.memory.set_readonly(path, readonly).await
    }
}

fn request(session: &RunMatSession, operation: TextOperation) -> crate::abi::ExecutionRequest {
    crate::abi::ExecutionRequest::for_source(
        crate::abi::SourceInput::Text {
            name: "<textlines-owner-cancellation-test>".into(),
            text: operation.source().into(),
        },
        session.compat_mode(),
        crate::abi::HostExecutionPolicy::default(),
        session.workspace_handle(),
    )
}

#[test]
fn pending_textline_operations_retain_selected_owner_and_cancel_without_replay() {
    let _lock = runmat_filesystem::provider_override_lock();
    for operation in [TextOperation::Read, TextOperation::Write] {
        let selected = Arc::new(TextProvider::new(Some(operation)));
        let evidence = selected.evidence.clone();
        let selected_memory = selected.memory.clone();
        let selected_weak = Arc::downgrade(&selected);
        let _restore = runmat_filesystem::replace_provider(selected.clone());
        runmat_filesystem::set_current_dir("/textlines").expect("selected virtual directory");

        let mut session = RunMatSession::with_options(false, false).expect("session");
        session
            .install_application_settings(Rc::new(
                runmat_runtime::context::IsolatedMemoryPreferenceService::new(),
            ))
            .expect("isolated application settings");
        let cancellation = session.interrupt_handle();
        let request = request(&session, operation);
        let mut call = Box::pin(session.execute_request(request));

        for _ in 0..32 {
            let mut context = Context::from_waker(noop_waker_ref());
            if let Poll::Ready(response) = call.as_mut().poll(&mut context) {
                panic!("text operation completed before reaching Pending: {response:?}");
            }
            if evidence.calls(operation) == 1 {
                break;
            }
        }
        assert_eq!(evidence.calls(operation), 1);
        assert_eq!(
            evidence.reads.load(Ordering::SeqCst),
            usize::from(operation == TextOperation::Read)
        );
        assert_eq!(
            evidence.writes.load(Ordering::SeqCst),
            usize::from(operation == TextOperation::Write)
        );
        assert_eq!(evidence.completed.load(Ordering::SeqCst), 0);

        let replacement = Arc::new(TextProvider::new(None));
        let replacement_evidence = replacement.evidence.clone();
        runmat_filesystem::set_provider(replacement.clone());
        drop(selected);
        assert!(
            selected_weak.upgrade().is_some(),
            "the Pending operation retains its selected provider"
        );

        cancellation.cancel(runmat_execution::CancellationReason::User);
        let response = futures::executor::block_on(call);
        let outcome = response
            .result
            .expect("source execution reports cancellation through outcome diagnostics");
        assert_eq!(outcome.diagnostics.len(), 1);
        assert_eq!(outcome.diagnostics[0].code, "RunMat:ExecutionCancelled");
        assert_eq!(
            outcome.diagnostics[0].severity,
            crate::abi::DiagnosticSeverity::Error
        );
        assert!(outcome.workspace_delta.upserts.is_empty());
        assert!(outcome.effects.is_empty());
        assert_eq!(evidence.calls(operation), 1, "operation was replayed");
        assert_eq!(evidence.retired.load(Ordering::SeqCst), 1);
        assert_eq!(evidence.completed.load(Ordering::SeqCst), 0);
        assert!(
            selected_memory
                .read_project_path("/textlines/lines.txt")
                .is_err(),
            "cancelled operation must not publish through the selected provider"
        );
        assert_eq!(replacement_evidence.reads.load(Ordering::SeqCst), 0);
        assert_eq!(replacement_evidence.writes.load(Ordering::SeqCst), 0);
        assert!(
            replacement
                .memory
                .read_project_path("/textlines/lines.txt")
                .is_err(),
            "cancelled invocation must not publish through the replacement"
        );
        // The session's startup search-path authority also owns the provider.
        // Pending retirement is checked above; final owner retirement follows
        // retirement of that independent session authority.
        drop(session);
        assert!(selected_weak.upgrade().is_none());
    }
}
