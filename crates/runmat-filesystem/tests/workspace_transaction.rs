use runmat_filesystem::{
    workspace::WorkspacePublicationOutcome, workspace_access::WorkspaceTransaction, FsProvider,
    MemoryFsProvider,
};
use std::{path::Path, sync::Arc};

#[test]
fn asynchronous_transaction_keeps_private_changes_invisible_and_reopens_results() {
    futures::executor::block_on(async {
        let provider = Arc::new(MemoryFsProvider::new());
        provider
            .create_dir_all(Path::new("/results"))
            .await
            .unwrap();
        provider
            .write(Path::new("/results/old"), b"old")
            .await
            .unwrap();
        let transaction =
            WorkspaceTransaction::capture(provider.clone(), "/results".into(), true, 4096)
                .await
                .unwrap();
        let scope = transaction
            .view()
            .workspace_access(Path::new("/results"), true, 4096)
            .unwrap();
        scope
            .atomic_create_new(Path::new("new"), b"complete", 4096)
            .unwrap();
        scope.remove_file(Path::new("old")).unwrap();
        assert!(provider.read(Path::new("/results/new")).await.is_err());
        assert_eq!(
            provider.read(Path::new("/results/old")).await.unwrap(),
            b"old"
        );
        assert!(matches!(
            transaction.commit().await.unwrap(),
            Some(WorkspacePublicationOutcome::Published { .. })
        ));
        let reopened = WorkspaceTransaction::capture(provider, "/results".into(), false, 4096)
            .await
            .unwrap();
        let scope = reopened
            .view()
            .workspace_access(Path::new("/results"), false, 4096)
            .unwrap();
        assert_eq!(scope.read(Path::new("new"), 4096).unwrap(), b"complete");
        assert!(scope.metadata(Path::new("old")).is_err());
        assert!(scope.read(Path::new("../outside"), 4096).is_err());
        assert!(scope
            .atomic_create_new(Path::new("denied"), b"x", 4096)
            .is_err());
        assert!(scope.read(Path::new("new"), 1).is_err());
    });
}

#[test]
fn concurrent_changes_conflict_without_exposing_any_staged_file() {
    futures::executor::block_on(async {
        let provider = Arc::new(MemoryFsProvider::new());
        let transaction =
            WorkspaceTransaction::capture(provider.clone(), "/results".into(), true, 4096)
                .await
                .unwrap();
        transaction
            .view()
            .workspace_access(Path::new("/results"), true, 4096)
            .unwrap()
            .atomic_create_new(Path::new("private"), b"never visible", 4096)
            .unwrap();
        provider
            .write(Path::new("/results/concurrent"), b"other writer")
            .await
            .unwrap();
        assert!(matches!(
            transaction.commit().await.unwrap(),
            Some(WorkspacePublicationOutcome::Conflict { .. })
        ));
        assert!(provider.read(Path::new("/results/private")).await.is_err());
        assert_eq!(
            provider
                .read(Path::new("/results/concurrent"))
                .await
                .unwrap(),
            b"other writer"
        );
    });
}

#[test]
fn dropping_an_operation_aborts_and_capture_is_bounded() {
    futures::executor::block_on(async {
        let provider = Arc::new(MemoryFsProvider::new());
        let transaction =
            WorkspaceTransaction::capture(provider.clone(), "/results".into(), true, 4096)
                .await
                .unwrap();
        transaction
            .view()
            .workspace_access(Path::new("/results"), true, 4096)
            .unwrap()
            .atomic_create_new(Path::new("private"), b"discard", 4096)
            .unwrap();
        drop(transaction);
        assert!(provider.read(Path::new("/results/private")).await.is_err());
        provider
            .write(Path::new("/results/large"), &[0; 32])
            .await
            .unwrap();
        assert!(
            WorkspaceTransaction::capture(provider, "/results".into(), false, 16)
                .await
                .is_err()
        );
    });
}

// Simulate a transport losing the acknowledgement after the authority commits.
struct LostAcknowledgement {
    base: MemoryFsProvider,
}
#[async_trait::async_trait(?Send)]
impl FsProvider for LostAcknowledgement {
    fn supports_workspace_transactions(&self) -> bool {
        true
    }
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        runmat_filesystem::FsPathNamespace::VirtualRooted
    }
    fn open(
        &self,
        path: &Path,
        flags: &runmat_filesystem::OpenFlags,
    ) -> std::io::Result<Box<dyn runmat_filesystem::FileHandle>> {
        self.base.open(path, flags)
    }
    async fn capture_workspace(
        &self,
        request: runmat_filesystem::workspace::WorkspaceCaptureRequest,
    ) -> std::io::Result<runmat_filesystem::workspace::WorkspaceSnapshot> {
        self.base.capture_workspace(request).await
    }
    async fn publish_workspace(
        &self,
        request: runmat_filesystem::workspace::WorkspacePublicationRequest,
    ) -> std::io::Result<WorkspacePublicationOutcome> {
        self.base.publish_workspace(request).await?;
        Err(std::io::Error::other("acknowledgement lost"))
    }
    async fn lookup_workspace_publication(
        &self,
        request: &runmat_filesystem::workspace::WorkspacePublicationRequest,
    ) -> std::io::Result<Option<WorkspacePublicationOutcome>> {
        self.base.lookup_workspace_publication(request).await
    }
    async fn read(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        self.base.read(path).await
    }
    async fn write(&self, path: &Path, data: &[u8]) -> std::io::Result<()> {
        self.base.write(path, data).await
    }
    async fn remove_file(&self, path: &Path) -> std::io::Result<()> {
        self.base.remove_file(path).await
    }
    async fn metadata(&self, path: &Path) -> std::io::Result<runmat_filesystem::FsMetadata> {
        self.base.metadata(path).await
    }
    async fn symlink_metadata(
        &self,
        path: &Path,
    ) -> std::io::Result<runmat_filesystem::FsMetadata> {
        self.base.symlink_metadata(path).await
    }
    async fn read_dir(&self, path: &Path) -> std::io::Result<Vec<runmat_filesystem::DirEntry>> {
        self.base.read_dir(path).await
    }
    async fn canonicalize(&self, path: &Path) -> std::io::Result<std::path::PathBuf> {
        self.base.canonicalize(path).await
    }
    async fn create_dir(&self, path: &Path) -> std::io::Result<()> {
        self.base.create_dir(path).await
    }
    async fn create_dir_all(&self, path: &Path) -> std::io::Result<()> {
        self.base.create_dir_all(path).await
    }
    async fn remove_dir(&self, path: &Path) -> std::io::Result<()> {
        self.base.remove_dir(path).await
    }
    async fn remove_dir_all(&self, path: &Path) -> std::io::Result<()> {
        self.base.remove_dir_all(path).await
    }
    async fn rename(&self, from: &Path, to: &Path) -> std::io::Result<()> {
        self.base.rename(from, to).await
    }
    async fn set_readonly(&self, path: &Path, readonly: bool) -> std::io::Result<()> {
        self.base.set_readonly(path, readonly).await
    }
}
#[test]
fn lost_acknowledgement_recovers_the_original_receipt_without_replaying_edits() {
    futures::executor::block_on(async {
        let provider = Arc::new(LostAcknowledgement {
            base: MemoryFsProvider::new(),
        });
        let transaction =
            WorkspaceTransaction::capture(provider.clone(), "/results".into(), true, 4096)
                .await
                .unwrap();
        transaction
            .view()
            .workspace_access(Path::new("/results"), true, 4096)
            .unwrap()
            .atomic_create_new(Path::new("saved"), b"once", 4096)
            .unwrap();
        assert!(matches!(
            transaction.commit().await.unwrap(),
            Some(WorkspacePublicationOutcome::Published { .. })
        ));
        assert_eq!(
            provider.read(Path::new("/results/saved")).await.unwrap(),
            b"once"
        );
    });
}
