//! Read/write attenuation for one already-authenticated selected provider.
//! This wrapper never creates authority: the native host constructs it only
//! after joining the exact reached effect issuers to its live provider.

#[cfg(not(target_arch = "wasm32"))]
use crate::NativePathIntent;
use crate::{
    data_contract::{
        DataChunkUploadRequest, DataChunkUploadTarget, DataManifestDescriptor, DataManifestRequest,
    },
    execution, workspace, workspace_access, AtomicCreateResult, AtomicReplaceOutcome, DirEntry,
    FileHandle, FsBoundedInput, FsManagedReadContract, FsMappedBacking, FsMetadata,
    FsPathNamespace, FsProvider, FsResourceLease, FsResourceLeaseHandle, FsStagedOutput, OpenFlags,
    ReadManyEntry,
};
use async_trait::async_trait;
use std::{
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

pub struct SelectedFsProvider {
    inner: Arc<dyn FsProvider>,
    read: bool,
    write: bool,
}

impl SelectedFsProvider {
    pub fn new(inner: Arc<dyn FsProvider>, read: bool, write: bool) -> io::Result<Self> {
        if !read && !write {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "selected filesystem has no admitted operation",
            ));
        }
        Ok(Self { inner, read, write })
    }

    fn require_read(&self) -> io::Result<()> {
        if self.read {
            Ok(())
        } else {
            Err(denied())
        }
    }

    fn require_write(&self) -> io::Result<()> {
        if self.write {
            Ok(())
        } else {
            Err(denied())
        }
    }

    fn require_absolute(&self, path: &Path) -> io::Result<()> {
        if self.inner.path_namespace().is_absolute(path) {
            Ok(())
        } else {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "selected filesystem operation must use an invocation-resolved absolute path",
            ))
        }
    }

    fn require_workspace_paths(&self, root: &Path, excluded: &[PathBuf]) -> io::Result<()> {
        self.require_absolute(root)?;
        for path in excluded {
            self.require_absolute(path)?;
        }
        Ok(())
    }
}

fn denied() -> io::Error {
    io::Error::new(
        io::ErrorKind::PermissionDenied,
        "selected filesystem effect is not admitted",
    )
}

#[async_trait(?Send)]
impl FsProvider for SelectedFsProvider {
    fn path_namespace(&self) -> FsPathNamespace {
        self.inner.path_namespace()
    }
    fn current_dir_override(&self) -> Option<PathBuf> {
        // InvocationFilesystem owns the selected working directory and gates
        // its visibility with EnvironmentRead. Never expose the underlying
        // provider's ambient/default directory through this attenuated owner.
        None
    }
    fn supports_read(&self) -> bool {
        self.read && self.inner.supports_read()
    }

    fn workspace_access(
        &self,
        root: &Path,
        writable: bool,
        maximum_resident_bytes: u64,
    ) -> io::Result<Box<dyn workspace_access::WorkspaceAccess>> {
        self.require_absolute(root)?;
        if writable {
            self.require_read()?;
            self.require_write()?;
        } else {
            self.require_read()?;
        }
        self.inner
            .workspace_access(root, writable, maximum_resident_bytes)
    }
    async fn capture_workspace(
        &self,
        request: workspace::WorkspaceCaptureRequest,
    ) -> io::Result<workspace::WorkspaceSnapshot> {
        self.require_read()?;
        self.require_workspace_paths(&request.root, &request.excluded_subtrees)?;
        self.inner.capture_workspace(request).await
    }
    async fn lookup_workspace_publication(
        &self,
        request: &workspace::WorkspacePublicationRequest,
    ) -> io::Result<Option<workspace::WorkspacePublicationOutcome>> {
        self.require_read()?;
        self.require_workspace_paths(&request.root, &request.excluded_subtrees)?;
        self.inner.lookup_workspace_publication(request).await
    }
    async fn publish_workspace(
        &self,
        request: workspace::WorkspacePublicationRequest,
    ) -> io::Result<workspace::WorkspacePublicationOutcome> {
        self.require_write()?;
        self.require_workspace_paths(&request.root, &request.excluded_subtrees)?;
        for edit in &request.edits {
            match edit {
                workspace::WorkspaceEdit::Write { path, .. }
                | workspace::WorkspaceEdit::Remove { path } => self.require_absolute(path)?,
                workspace::WorkspaceEdit::Rename { from, to } => {
                    self.require_absolute(from)?;
                    self.require_absolute(to)?;
                }
            }
        }
        self.inner.publish_workspace(request).await
    }
    async fn transition_execution(
        &self,
        work: &execution::ExecutionWorkId,
        command: execution::ExecutionLeaseCommand,
    ) -> io::Result<execution::ExecutionLeaseOutcome> {
        self.require_write()?;
        self.inner.transition_execution(work, command).await
    }
    async fn publish_workspace_fenced(
        &self,
        request: workspace::WorkspacePublicationRequest,
        lease: &execution::ExecutionLease,
    ) -> io::Result<execution::FencedWorkspacePublicationOutcome> {
        self.require_write()?;
        self.require_workspace_paths(&request.root, &request.excluded_subtrees)?;
        for edit in &request.edits {
            match edit {
                workspace::WorkspaceEdit::Write { path, .. }
                | workspace::WorkspaceEdit::Remove { path } => self.require_absolute(path)?,
                workspace::WorkspaceEdit::Rename { from, to } => {
                    self.require_absolute(from)?;
                    self.require_absolute(to)?;
                }
            }
        }
        self.inner.publish_workspace_fenced(request, lease).await
    }
    async fn open_mapped_backing(
        &self,
        path: &Path,
        writable: bool,
    ) -> io::Result<Option<Box<dyn FsMappedBacking>>> {
        self.require_absolute(path)?;
        if writable {
            self.require_read()?;
            self.require_write()?;
        } else {
            self.require_read()?;
        }
        self.inner.open_mapped_backing(path, writable).await
    }
    fn begin_staged_replace(&self, path: &Path) -> io::Result<Option<Box<dyn FsStagedOutput>>> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.begin_staged_replace(path)
    }
    #[cfg(not(target_arch = "wasm32"))]
    fn acquire_native_path(
        &self,
        path: &Path,
        intent: NativePathIntent,
    ) -> io::Result<Option<crate::NativePathGrant>> {
        self.require_absolute(path)?;
        match intent {
            NativePathIntent::Read | NativePathIntent::ProcessWorkingDirectory => {
                self.require_read()?
            }
            NativePathIntent::ReadWriteCreate => {
                self.require_read()?;
                self.require_write()?;
            }
        }
        self.inner.acquire_native_path(path, intent)
    }
    async fn open_bounded_input(&self, path: &Path) -> io::Result<Option<FsBoundedInput>> {
        self.require_read()?;
        self.require_absolute(path)?;
        self.inner.open_bounded_input(path).await
    }
    fn managed_read_contract(&self) -> Option<FsManagedReadContract> {
        self.read
            .then(|| self.inner.managed_read_contract())
            .flatten()
    }
    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        (self.read || self.write)
            .then(|| self.inner.resource_lease())
            .flatten()
    }
    fn resource_lease_handle(&self) -> Option<FsResourceLeaseHandle> {
        (self.read || self.write)
            .then(|| self.inner.resource_lease_handle())
            .flatten()
    }
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.require_absolute(path)?;
        let writable =
            flags.write || flags.append || flags.truncate || flags.create || flags.create_new;
        if flags.read || !writable {
            self.require_read()?;
        }
        if writable {
            self.require_write()?;
        }
        self.inner.open(path, flags)
    }
    async fn open_async(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.require_absolute(path)?;
        let writable =
            flags.write || flags.append || flags.truncate || flags.create || flags.create_new;
        if flags.read || !writable {
            self.require_read()?;
        }
        if writable {
            self.require_write()?;
        }
        self.inner.open_async(path, flags).await
    }
    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        self.require_read()?;
        self.require_absolute(path)?;
        self.inner.read(path).await
    }
    async fn read_version(
        &self,
        path: &Path,
        version: &str,
        size: u64,
        snapshot: &str,
    ) -> io::Result<Vec<u8>> {
        self.require_read()?;
        self.require_absolute(path)?;
        self.inner.read_version(path, version, size, snapshot).await
    }
    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.write(path, data).await
    }
    async fn publish_output(
        &self,
        path: &Path,
        data: &[u8],
        expected: Option<&str>,
    ) -> io::Result<String> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.publish_output(path, data, expected).await
    }
    async fn remove_output(&self, path: &Path, expected: &str) -> io::Result<()> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.remove_output(path, expected).await
    }
    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.remove_file(path).await
    }
    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.require_read()?;
        self.require_absolute(path)?;
        self.inner.metadata(path).await
    }
    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        self.require_read()?;
        self.require_absolute(path)?;
        self.inner.symlink_metadata(path).await
    }
    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        self.require_read()?;
        self.require_absolute(path)?;
        self.inner.read_dir(path).await
    }
    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        self.require_read()?;
        self.require_absolute(path)?;
        self.inner.canonicalize(path).await
    }
    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.create_dir(path).await
    }
    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.create_dir_all(path).await
    }
    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.remove_dir(path).await
    }
    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.remove_dir_all(path).await
    }
    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.require_write()?;
        self.require_absolute(from)?;
        self.require_absolute(to)?;
        self.inner.rename(from, to).await
    }
    async fn rename_replace(&self, from: &Path, to: &Path) -> io::Result<()> {
        self.require_write()?;
        self.require_absolute(from)?;
        self.require_absolute(to)?;
        self.inner.rename_replace(from, to).await
    }
    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.set_readonly(path, readonly).await
    }
    async fn atomic_create_new(&self, path: &Path, bytes: &[u8]) -> io::Result<AtomicCreateResult> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.atomic_create_new(path, bytes).await
    }
    async fn atomic_replace(&self, path: &Path, bytes: &[u8]) -> io::Result<AtomicReplaceOutcome> {
        self.require_write()?;
        self.require_absolute(path)?;
        self.inner.atomic_replace(path, bytes).await
    }
    async fn read_many(&self, paths: &[PathBuf]) -> io::Result<Vec<ReadManyEntry>> {
        self.require_read()?;
        for path in paths {
            self.require_absolute(path)?;
        }
        self.inner.read_many(paths).await
    }
    async fn data_manifest_descriptor(
        &self,
        _request: &DataManifestRequest,
    ) -> io::Result<DataManifestDescriptor> {
        self.require_read()?;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "dataset metadata requires a separate selected remote service",
        ))
    }
    async fn data_chunk_upload_targets(
        &self,
        _request: &DataChunkUploadRequest,
    ) -> io::Result<Vec<DataChunkUploadTarget>> {
        self.require_read()?;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "dataset upload targets require a separate selected remote service",
        ))
    }
    async fn data_upload_chunk(
        &self,
        _target: &DataChunkUploadTarget,
        _data: &[u8],
    ) -> io::Result<()> {
        self.require_write()?;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "dataset chunk upload requires a separate selected remote service",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_and_write_issuers_attenuate_the_actual_provider() {
        let memory = crate::MemoryFsProvider::new();
        memory.write_project_path("/sample", b"original").unwrap();
        let source: Arc<dyn FsProvider> = Arc::new(memory.clone());
        let read_only = SelectedFsProvider::new(Arc::clone(&source), true, false).unwrap();
        assert_eq!(
            futures::executor::block_on(read_only.read(Path::new("/sample"))).unwrap(),
            b"original"
        );
        assert_eq!(
            futures::executor::block_on(read_only.write(Path::new("/sample"), b"changed"))
                .unwrap_err()
                .kind(),
            io::ErrorKind::PermissionDenied
        );
        assert_eq!(memory.read_project_path("/sample").unwrap(), b"original");

        let write_only = SelectedFsProvider::new(source, false, true).unwrap();
        assert_eq!(
            futures::executor::block_on(write_only.read(Path::new("/sample")))
                .unwrap_err()
                .kind(),
            io::ErrorKind::PermissionDenied
        );
        futures::executor::block_on(write_only.write(Path::new("/sample"), b"changed")).unwrap();
        assert_eq!(memory.read_project_path("/sample").unwrap(), b"changed");
    }

    #[test]
    fn write_flags_require_write_issuer_even_when_read_is_requested() {
        let source: Arc<dyn FsProvider> = Arc::new(crate::MemoryFsProvider::new());
        let read_only = SelectedFsProvider::new(source, true, false).unwrap();
        let flags = OpenFlags {
            read: true,
            create: true,
            ..OpenFlags::default()
        };
        assert_eq!(
            read_only
                .open(Path::new("/missing"), &flags)
                .err()
                .expect("write-including open must be rejected")
                .kind(),
            io::ErrorKind::PermissionDenied
        );
    }

    #[test]
    fn direct_relative_provider_calls_cannot_inherit_process_working_directory() {
        let memory = crate::MemoryFsProvider::new();
        memory.write_project_path("/input", b"original").unwrap();
        let selected = SelectedFsProvider::new(Arc::new(memory.clone()), true, true).unwrap();
        assert_eq!(
            futures::executor::block_on(selected.read(Path::new("input")))
                .unwrap_err()
                .kind(),
            io::ErrorKind::PermissionDenied
        );
        assert_eq!(
            futures::executor::block_on(selected.write(Path::new("input"), b"changed"))
                .unwrap_err()
                .kind(),
            io::ErrorKind::PermissionDenied
        );
        assert_eq!(memory.read_project_path("/input").unwrap(), b"original");
    }

    #[test]
    fn selected_provider_never_exposes_the_underlying_default_directory() {
        let source: Arc<dyn FsProvider> = Arc::new(crate::MemoryFsProvider::with_current_dir(
            PathBuf::from("/private"),
        ));
        let selected = SelectedFsProvider::new(source, true, false).unwrap();
        assert_eq!(selected.current_dir_override(), None);
    }
}
