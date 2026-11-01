#[cfg(not(target_arch = "wasm32"))]
use crate::data_contract::{
    DataChunkUploadRequest, DataChunkUploadTarget, DataManifestDescriptor, DataManifestRequest,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::{AtomicCreateResult, DirEntry, FsFileType, FsMetadata, FsProvider, OpenFlags};
#[cfg(not(target_arch = "wasm32"))]
use async_trait::async_trait;
#[cfg(not(target_arch = "wasm32"))]
use chrono::Utc;
#[cfg(not(target_arch = "wasm32"))]
use serde_json::Value as JsonValue;
#[cfg(not(target_arch = "wasm32"))]
use std::ffi::OsString;
#[cfg(not(target_arch = "wasm32"))]
use std::fs;
#[cfg(not(target_arch = "wasm32"))]
use std::io;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Component, Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
#[path = "sandbox/staged.rs"]
mod staged;

#[cfg(not(target_arch = "wasm32"))]
/// Filesystem provider that sandboxes all operations under a fixed root directory.
///
/// Incoming paths (absolute or relative) are normalized and resolved relative to the sandbox root.
/// Attempts to traverse outside the root using `..` simply clamp to the root, preventing escape.
pub struct SandboxFsProvider {
    root: PathBuf,
    captured_io: Option<Arc<dyn CapturedIo>>,
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait(?Send)]
pub trait CapturedIo: Send + Sync + 'static {
    async fn publish_workspace_fenced(
        &self,
        _request: crate::workspace::WorkspacePublicationRequest,
        _lease: &crate::execution::ExecutionLease,
    ) -> io::Result<crate::execution::FencedWorkspacePublicationOutcome> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "authoritative fenced publication is unavailable",
        ))
    }
    async fn transition_execution(
        &self,
        _work: &crate::execution::ExecutionWorkId,
        _command: crate::execution::ExecutionLeaseCommand,
    ) -> io::Result<crate::execution::ExecutionLeaseOutcome> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "authoritative execution ownership is unavailable",
        ))
    }
    async fn lookup_workspace_publication(
        &self,
        _request: &crate::workspace::WorkspacePublicationRequest,
    ) -> io::Result<Option<crate::workspace::WorkspacePublicationOutcome>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "authoritative workspace receipts are unavailable",
        ))
    }
    async fn capture_workspace(
        &self,
        _request: crate::workspace::WorkspaceCaptureRequest,
    ) -> io::Result<crate::workspace::WorkspaceSnapshot> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "authoritative workspace capture is unavailable",
        ))
    }
    async fn publish_workspace(
        &self,
        _request: crate::workspace::WorkspacePublicationRequest,
    ) -> io::Result<crate::workspace::WorkspacePublicationOutcome> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "authoritative workspace publication is unavailable",
        ))
    }
    async fn read_version(
        &self,
        path: &Path,
        version: &str,
        size: u64,
        snapshot: &str,
    ) -> io::Result<Vec<u8>>;
    async fn publish_output(
        &self,
        path: &Path,
        data: &[u8],
        expected: Option<&str>,
    ) -> io::Result<String>;
    async fn remove_output(&self, path: &Path, expected: &str) -> io::Result<()>;
}

#[cfg(not(target_arch = "wasm32"))]
impl SandboxFsProvider {
    /// Create a new sandbox rooted at `root`. The directory is created if it does not exist.
    pub fn new(root: PathBuf) -> io::Result<Self> {
        if !root.exists() {
            fs::create_dir_all(&root)?;
        }
        let canonical = fs::canonicalize(root)?;
        Ok(Self {
            root: canonical,
            captured_io: None,
        })
    }

    /// Keep ordinary project reads in the sandbox while captured runs use the
    /// authoritative provider for immutable versions and conditional outputs.
    pub fn with_captured_io(mut self, provider: Arc<dyn CapturedIo>) -> Self {
        self.captured_io = Some(provider);
        self
    }

    /// Return the sandbox root on the host filesystem.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn resolve(&self, path: &Path) -> PathBuf {
        let mut segments: Vec<OsString> = Vec::new();
        for component in path.components() {
            match component {
                Component::Prefix(_) | Component::RootDir => {
                    segments.clear();
                }
                Component::CurDir => {}
                Component::ParentDir => {
                    segments.pop();
                }
                Component::Normal(seg) => segments.push(seg.to_os_string()),
            }
        }
        let mut target = self.root.clone();
        for seg in segments {
            target.push(seg);
        }
        target
    }

    fn virtualize(&self, real: &Path) -> PathBuf {
        let relative = real.strip_prefix(&self.root).unwrap_or(Path::new(""));
        if relative.as_os_str().is_empty() {
            return PathBuf::from("/");
        }

        let mut path = String::from("/");
        path.push_str(&relative.to_string_lossy().replace('\\', "/"));
        PathBuf::from(path)
    }

    fn make_dir_entry(&self, real_path: PathBuf, file_name: OsString) -> DirEntry {
        let file_type = fs::symlink_metadata(&real_path)
            .ok()
            .map(|m| FsFileType::from(m.file_type()))
            .unwrap_or(FsFileType::Unknown);
        DirEntry {
            path: self.virtualize(&real_path),
            file_name,
            file_type,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait(?Send)]
impl FsProvider for SandboxFsProvider {
    fn supports_workspace_transactions(&self) -> bool {
        true
    }
    fn workspace_access(
        &self,
        root: &Path,
        writable: bool,
        maximum_resident_bytes: u64,
    ) -> io::Result<Box<dyn crate::workspace_access::WorkspaceAccess>> {
        if self.captured_io.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "captured authority does not expose coherent workspace access",
            ));
        }
        if root
            .components()
            .any(|part| matches!(part, Component::ParentDir | Component::Prefix(_)))
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "workspace access root escapes its mount",
            ));
        }
        let target = self.resolve(root);
        let mount_guard = if target != self.root {
            crate::workspace_native::guard_admitted(&self.root, true, maximum_resident_bytes)?
        } else {
            None
        };
        Ok(Box::new(crate::workspace_access::NativeAccess::open(
            target,
            writable,
            mount_guard,
            maximum_resident_bytes,
        )?))
    }
    async fn publish_workspace_fenced(
        &self,
        request: crate::workspace::WorkspacePublicationRequest,
        lease: &crate::execution::ExecutionLease,
    ) -> io::Result<crate::execution::FencedWorkspacePublicationOutcome> {
        if let Some(authority) = &self.captured_io {
            return authority.publish_workspace_fenced(request, lease).await;
        }
        crate::execution::fenced_outcome(
            crate::workspace_native::publish_with_lease(&self.root, request, Some(lease)).await,
        )
    }
    async fn transition_execution(
        &self,
        work: &crate::execution::ExecutionWorkId,
        command: crate::execution::ExecutionLeaseCommand,
    ) -> io::Result<crate::execution::ExecutionLeaseOutcome> {
        if let Some(authority) = &self.captured_io {
            return authority.transition_execution(work, command).await;
        }
        crate::workspace_native::transition_execution(&self.root, work, command)
    }
    async fn lookup_workspace_publication(
        &self,
        request: &crate::workspace::WorkspacePublicationRequest,
    ) -> io::Result<Option<crate::workspace::WorkspacePublicationOutcome>> {
        if let Some(authority) = &self.captured_io {
            return authority.lookup_workspace_publication(request).await;
        }
        crate::workspace_native::lookup(&self.root, request)
    }
    async fn capture_workspace(
        &self,
        request: crate::workspace::WorkspaceCaptureRequest,
    ) -> io::Result<crate::workspace::WorkspaceSnapshot> {
        if let Some(authority) = &self.captured_io {
            return authority.capture_workspace(request).await;
        }
        crate::workspace_native::capture(&self.root, request)
    }

    async fn publish_workspace(
        &self,
        request: crate::workspace::WorkspacePublicationRequest,
    ) -> io::Result<crate::workspace::WorkspacePublicationOutcome> {
        if let Some(authority) = &self.captured_io {
            return authority.publish_workspace(request).await;
        }
        crate::workspace_native::publish(&self.root, request).await
    }
    fn path_namespace(&self) -> crate::FsPathNamespace {
        crate::FsPathNamespace::VirtualRooted
    }

    fn current_dir_override(&self) -> Option<PathBuf> {
        Some(PathBuf::from("/"))
    }

    fn begin_staged_replace(
        &self,
        destination: &Path,
    ) -> io::Result<Option<Box<dyn crate::FsStagedOutput>>> {
        // Sandbox paths are virtual-rooted (`/reports/out.csv`), including on
        // Windows where that spelling has a root but no drive prefix.
        if !self.path_namespace().is_absolute(destination) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sandbox staged destination must be an absolute virtual path",
            ));
        }
        let resolved = self.resolve(destination);
        let authorized = staged::authorized_destination(&self.root, &resolved)?;
        crate::native::staged::begin(&authorized).map(Some)
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn crate::FileHandle>> {
        let target = self.resolve(path);
        let writable =
            flags.write || flags.append || flags.truncate || flags.create || flags.create_new;
        let _guard = crate::workspace_native::guard(&self.root, writable)?;
        if writable {
            crate::workspace_native::protected(&self.root, &target)?;
        } else {
            crate::workspace_native::ensure_no_symlink(&self.root, &target)?;
        }
        if writable {
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut opts = fs::OpenOptions::new();
        opts.read(flags.read);
        opts.write(flags.write);
        opts.append(flags.append);
        opts.truncate(flags.truncate);
        opts.create(flags.create);
        opts.create_new(flags.create_new);
        let file = opts.open(&target)?;
        if writable {
            Ok(Box::new(crate::workspace_native::WritableHandle {
                file,
                root: self.root.clone(),
                path: target,
            }))
        } else {
            Ok(Box::new(file))
        }
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let _guard = crate::workspace_native::guard(&self.root, false)?;
        let target = self.resolve(path);
        crate::workspace_native::ensure_no_symlink(&self.root, &target)?;
        fs::read(target)
    }

    async fn read_version(
        &self,
        path: &Path,
        version: &str,
        size: u64,
        snapshot: &str,
    ) -> io::Result<Vec<u8>> {
        let provider = self.captured_io.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "immutable version reads are unsupported by this provider",
            )
        })?;
        provider.read_version(path, version, size, snapshot).await
    }

    async fn publish_output(
        &self,
        path: &Path,
        data: &[u8],
        expected: Option<&str>,
    ) -> io::Result<String> {
        let provider = self.captured_io.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "conditional output publication is unsupported",
            )
        })?;
        provider.publish_output(path, data, expected).await
    }

    async fn remove_output(&self, path: &Path, expected: &str) -> io::Result<()> {
        let provider = self.captured_io.as_ref().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "conditional output removal is unsupported",
            )
        })?;
        provider.remove_output(path, expected).await
    }

    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let _guard = crate::workspace_native::guard(&self.root, true)?;
        let target = self.resolve(path);
        crate::workspace_native::protected(&self.root, &target)?;
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(target, data)
    }

    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        let _guard = crate::workspace_native::guard(&self.root, true)?;
        let target = self.resolve(path);
        crate::workspace_native::protected(&self.root, &target)?;
        if target.exists() {
            fs::remove_file(target)?;
        }
        Ok(())
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        let _guard = crate::workspace_native::guard(&self.root, false)?;
        let target = self.resolve(path);
        crate::workspace_native::ensure_no_symlink(&self.root, &target)?;
        fs::metadata(target).map(FsMetadata::from)
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        let _guard = crate::workspace_native::guard(&self.root, false)?;
        let target = self.resolve(path);
        if let Some(parent) = target
            .parent()
            .filter(|parent| parent.starts_with(&self.root))
        {
            crate::workspace_native::ensure_no_symlink(&self.root, parent)?;
        }
        fs::symlink_metadata(target).map(FsMetadata::from)
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let _guard = crate::workspace_native::guard(&self.root, false)?;
        let target = self.resolve(path);
        crate::workspace_native::ensure_no_symlink(&self.root, &target)?;
        let entries = fs::read_dir(&target)?;
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry?;
            out.push(self.make_dir_entry(entry.path(), entry.file_name()));
        }
        Ok(out)
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        let _guard = crate::workspace_native::guard(&self.root, false)?;
        let target = self.resolve(path);
        let real = fs::canonicalize(target)?;
        if !real.starts_with(&self.root) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "path escapes the filesystem sandbox",
            ));
        }
        Ok(self.virtualize(&real))
    }

    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        let _guard = crate::workspace_native::guard(&self.root, true)?;
        let target = self.resolve(path);
        crate::workspace_native::protected(&self.root, &target)?;
        fs::create_dir(&target)
    }

    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        let _guard = crate::workspace_native::guard(&self.root, true)?;
        let target = self.resolve(path);
        if target.is_dir() {
            crate::workspace_native::ensure_no_symlink(&self.root, &target)?;
            return Ok(());
        }
        crate::workspace_native::protected(&self.root, &target)?;
        fs::create_dir_all(&target)
    }

    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        let _guard = crate::workspace_native::guard(&self.root, true)?;
        let target = self.resolve(path);
        crate::workspace_native::protected(&self.root, &target)?;
        fs::remove_dir(&target)
    }

    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        let _guard = crate::workspace_native::guard(&self.root, true)?;
        let target = self.resolve(path);
        crate::workspace_native::protected(&self.root, &target)?;
        if target.exists() {
            fs::remove_dir_all(&target)?;
        }
        Ok(())
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let _guard = crate::workspace_native::guard(&self.root, true)?;
        let src = self.resolve(from);
        let dst = self.resolve(to);
        crate::workspace_native::protected(&self.root, &src)?;
        crate::workspace_native::protected(&self.root, &dst)?;
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::rename(src, dst)
    }

    async fn rename_replace(&self, from: &Path, to: &Path) -> io::Result<()> {
        let _guard = crate::workspace_native::guard(&self.root, true)?;
        let source = self.resolve(from);
        let destination = self.resolve(to);
        crate::workspace_native::protected(&self.root, &source)?;
        crate::workspace_native::protected(&self.root, &destination)?;
        crate::native::rename::replace(&source, &destination)
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        let _guard = crate::workspace_native::guard(&self.root, true)?;
        let target = self.resolve(path);
        crate::workspace_native::protected(&self.root, &target)?;
        let mut perms = fs::metadata(&target)?.permissions();
        perms.set_readonly(readonly);
        fs::set_permissions(target, perms)
    }

    async fn atomic_create_new(&self, path: &Path, bytes: &[u8]) -> io::Result<AtomicCreateResult> {
        let _guard = crate::workspace_native::guard(&self.root, true)?;
        let target = self.resolve(path);
        crate::workspace_native::protected(&self.root, &target)?;
        let parent = target.parent().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "atomic publication path has no parent directory",
            )
        })?;
        fs::create_dir_all(parent)?;
        let canonical_parent = fs::canonicalize(parent)?;
        if !canonical_parent.starts_with(&self.root) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "atomic publication parent escapes the filesystem sandbox",
            ));
        }
        crate::native::atomic_create_new_native(&target, bytes)
    }

    async fn data_manifest_descriptor(
        &self,
        request: &DataManifestRequest,
    ) -> io::Result<DataManifestDescriptor> {
        let manifest_path = if request.path.ends_with(".json") {
            PathBuf::from(&request.path)
        } else {
            PathBuf::from(&request.path).join("manifest.json")
        };
        let bytes = self.read(&manifest_path).await?;
        let json: JsonValue = serde_json::from_slice(&bytes)
            .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err.to_string()))?;
        Ok(DataManifestDescriptor {
            schema_version: json
                .get("schema_version")
                .or_else(|| json.get("schemaVersion"))
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as u32,
            format: json
                .get("format")
                .and_then(|v| v.as_str())
                .unwrap_or("runmat-data")
                .to_string(),
            dataset_id: json
                .get("dataset_id")
                .or_else(|| json.get("datasetId"))
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            updated_at: json
                .get("updated_at")
                .or_else(|| json.get("updatedAt"))
                .and_then(|v| v.as_str())
                .map(ToString::to_string)
                .unwrap_or_else(|| Utc::now().to_rfc3339()),
            txn_sequence: json
                .get("txn_sequence")
                .or_else(|| json.get("txnSequence"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
        })
    }

    async fn data_chunk_upload_targets(
        &self,
        request: &DataChunkUploadRequest,
    ) -> io::Result<Vec<DataChunkUploadTarget>> {
        let root = PathBuf::from(&request.dataset_path)
            .join("arrays")
            .join(sanitize_segment(&request.array))
            .join("chunks");
        self.create_dir_all(&root).await?;
        request
            .chunks
            .iter()
            .map(|chunk| {
                let path = root.join(format!("{}.bin", sanitize_segment(&chunk.object_id)));
                Ok(DataChunkUploadTarget {
                    key: chunk.key.clone(),
                    method: "PUT".to_string(),
                    upload_url: format!("sandbox://{}", path.to_string_lossy()),
                    headers: std::collections::HashMap::new(),
                })
            })
            .collect()
    }

    async fn data_upload_chunk(
        &self,
        target: &DataChunkUploadTarget,
        data: &[u8],
    ) -> io::Result<()> {
        if !target.method.eq_ignore_ascii_case("PUT") {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("unsupported upload method '{}'", target.method),
            ));
        }
        let path = target
            .upload_url
            .strip_prefix("sandbox://")
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "invalid sandbox upload url")
            })?;
        self.write(Path::new(path), data).await
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn sanitize_segment(input: &str) -> String {
    input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(all(not(target_arch = "wasm32"), test))]
mod tests {
    use super::{CapturedIo, SandboxFsProvider};
    use crate::FsProvider;
    use futures::executor;
    use std::fs;
    use std::io::ErrorKind;
    use std::path::Path;
    use std::{io, sync::Arc};
    use tempfile::tempdir;

    fn virtual_path(path: &Path) -> String {
        path.to_string_lossy().replace('\\', "/")
    }

    struct CapturedFixture;

    #[async_trait::async_trait(?Send)]
    impl CapturedIo for CapturedFixture {
        async fn read_version(
            &self,
            _path: &Path,
            version: &str,
            size: u64,
            snapshot: &str,
        ) -> io::Result<Vec<u8>> {
            if version != "v1" || snapshot != "snapshot" || size != 3 {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    "version unavailable",
                ));
            }
            Ok(b"old".to_vec())
        }
        async fn publish_output(
            &self,
            _path: &Path,
            _data: &[u8],
            _expected: Option<&str>,
        ) -> io::Result<String> {
            Ok("published".to_owned())
        }
        async fn remove_output(&self, _path: &Path, _expected: &str) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn execution_authority_never_falls_back_to_the_cache_for_a_remote_project() {
        let temp = tempdir().unwrap();
        let provider = SandboxFsProvider::new(temp.path().to_path_buf())
            .unwrap()
            .with_captured_io(Arc::new(CapturedFixture));
        let work = crate::execution::ExecutionWorkId {
            session_id: crate::execution::SessionId("session".into()),
            turn_id: crate::execution::TurnId("turn".into()),
        };
        let error = executor::block_on(provider.transition_execution(
            &work,
            crate::execution::ExecutionLeaseCommand::Acquire {
                owner: crate::execution::ExecutionOwnerId([1; 16]),
                duration_ms: 30_000,
            },
        ))
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::Unsupported);
        assert!(!temp
            .path()
            .join(".artifacts/workspace-publications")
            .exists());
    }

    #[test]
    fn cached_live_file_cannot_substitute_for_a_captured_version() {
        let temp = tempdir().expect("tempdir");
        let provider = SandboxFsProvider::new(temp.path().to_path_buf())
            .expect("sandbox")
            .with_captured_io(Arc::new(CapturedFixture));
        executor::block_on(provider.write(Path::new("/basic.m"), b"new")).expect("cache write");
        assert_eq!(
            executor::block_on(provider.read(Path::new("/basic.m"))).unwrap(),
            b"new"
        );
        assert_eq!(
            executor::block_on(provider.read_version(Path::new("/basic.m"), "v1", 3, "snapshot"))
                .unwrap(),
            b"old"
        );
        assert_eq!(
            executor::block_on(provider.read_version(
                Path::new("/basic.m"),
                "missing",
                3,
                "snapshot"
            ))
            .unwrap_err()
            .kind(),
            io::ErrorKind::NotFound
        );
        assert_eq!(
            executor::block_on(provider.publish_output(Path::new("/output"), b"data", None))
                .unwrap(),
            "published"
        );
    }

    #[test]
    fn sandbox_prevents_root_escape_and_virtualizes_paths() {
        let temp = tempdir().expect("tempdir");
        let provider = SandboxFsProvider::new(temp.path().to_path_buf()).expect("sandbox");
        executor::block_on(provider.create_dir_all(Path::new("nested/sub"))).expect("create dir");
        executor::block_on(provider.write(Path::new("nested/sub/file.txt"), b"hello"))
            .expect("write");

        // Attempt to escape root should clamp to sandbox.
        executor::block_on(provider.write(Path::new("../evil.txt"), b"nope"))
            .expect("write outside clamped");
        let entries = executor::block_on(provider.read_dir(Path::new("."))).expect("read root");
        assert!(entries.iter().any(|entry| entry.file_name() == "evil.txt"));

        let listing =
            executor::block_on(provider.read_dir(Path::new("nested"))).expect("list nested");
        assert!(listing.iter().any(|entry| {
            let path = virtual_path(entry.path());
            path == "/nested/sub"
        }));

        let sandbox_read =
            executor::block_on(provider.read(Path::new("/nested/sub/file.txt"))).expect("vfs read");
        assert_eq!(sandbox_read, b"hello");
    }

    #[test]
    fn canonicalize_returns_virtual_paths() {
        let temp = tempdir().expect("tempdir");
        let provider = SandboxFsProvider::new(temp.path().to_path_buf()).expect("sandbox");
        executor::block_on(provider.create_dir_all(Path::new("data"))).expect("create dir");
        executor::block_on(provider.write(Path::new("data/file.bin"), b"bytes")).expect("write");
        let canonical = executor::block_on(provider.canonicalize(Path::new("./data/./file.bin")))
            .expect("canonicalize");
        assert_eq!(virtual_path(&canonical), "/data/file.bin");
    }

    #[cfg(unix)]
    #[test]
    fn atomic_publication_rejects_a_symlinked_parent_escape() {
        use std::os::unix::fs::symlink;

        let sandbox_root = tempdir().expect("sandbox root");
        let outside = tempdir().expect("outside root");
        symlink(outside.path(), sandbox_root.path().join("escape")).unwrap();
        let provider =
            SandboxFsProvider::new(sandbox_root.path().to_path_buf()).expect("sandbox provider");

        let error = executor::block_on(
            provider.atomic_create_new(Path::new("/escape/artifact"), b"payload"),
        )
        .unwrap_err();

        assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied);
        assert!(!outside.path().join("artifact").exists());
    }

    #[test]
    fn staged_replacement_aborts_and_then_publishes_only_inside_captured_root() {
        let selected = tempdir().expect("selected root");
        let other = tempdir().expect("other root");
        let provider = SandboxFsProvider::new(selected.path().to_path_buf()).expect("sandbox");
        let other_provider = SandboxFsProvider::new(other.path().to_path_buf()).expect("other");
        let virtual_destination = Path::new("/reports/output.csv");
        executor::block_on(provider.write(virtual_destination, b"original"))
            .expect("existing destination");
        executor::block_on(other_provider.write(virtual_destination, b"other"))
            .expect("other destination");

        let mut abandoned = provider
            .begin_staged_replace(virtual_destination)
            .expect("acquire")
            .expect("sandbox stage");
        assert_eq!(
            executor::block_on(abandoned.write_chunk(b"aborted")).unwrap(),
            7
        );
        drop(abandoned);
        assert_eq!(
            executor::block_on(provider.read(virtual_destination)).unwrap(),
            b"original"
        );
        assert_eq!(
            fs::read_dir(selected.path().join("reports"))
                .expect("stage parent")
                .count(),
            1,
            "abandoned stage leaves only the original destination"
        );

        let mut stage = provider
            .begin_staged_replace(virtual_destination)
            .expect("acquire")
            .expect("sandbox stage");
        assert_eq!(
            executor::block_on(stage.write_chunk(b"replacement")).unwrap(),
            11
        );
        executor::block_on(stage.publish()).expect("publish");
        assert_eq!(
            executor::block_on(provider.read(virtual_destination)).unwrap(),
            b"replacement"
        );
        assert_eq!(
            executor::block_on(other_provider.read(virtual_destination)).unwrap(),
            b"other",
            "stage retains its selected sandbox root"
        );
    }

    #[test]
    fn staged_replacement_rejects_root_and_failed_publication_preserves_target() {
        let root = tempdir().expect("root");
        let provider = SandboxFsProvider::new(root.path().to_path_buf()).expect("sandbox");
        assert_eq!(
            provider
                .begin_staged_replace(Path::new("/"))
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            provider
                .begin_staged_replace(Path::new("relative.csv"))
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidInput
        );
        executor::block_on(provider.create_dir(Path::new("/directory-target")))
            .expect("directory target");
        let mut stage = provider
            .begin_staged_replace(Path::new("/directory-target"))
            .expect("stage")
            .expect("sandbox capability");
        executor::block_on(stage.write_chunk(b"replacement")).expect("write");
        assert!(executor::block_on(stage.publish()).is_err());
        assert!(root.path().join("directory-target").is_dir());
    }

    #[cfg(windows)]
    #[test]
    fn staged_replacement_rejects_native_windows_namespaces() {
        let root = tempdir().expect("root");
        let provider = SandboxFsProvider::new(root.path().to_path_buf()).expect("sandbox");
        for path in [r"C:\reports\output.csv", r"\\server\share\output.csv"] {
            let error = provider
                .begin_staged_replace(Path::new(path))
                .err()
                .expect("native namespace is not a virtual destination");
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
        }
        assert_eq!(fs::read_dir(root.path()).expect("sandbox root").count(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn staged_replacement_rejects_existing_symlink_escape_without_touching_target() {
        use std::os::unix::fs::symlink;

        let root = tempdir().expect("root");
        let outside = tempdir().expect("outside");
        fs::write(outside.path().join("output.csv"), b"outside").expect("outside sentinel");
        symlink(outside.path(), root.path().join("escape")).expect("symlink fixture");
        let provider = SandboxFsProvider::new(root.path().to_path_buf()).expect("sandbox");
        let error = provider
            .begin_staged_replace(Path::new("/escape/output.csv"))
            .err()
            .expect("escape rejected");
        assert_eq!(error.kind(), ErrorKind::PermissionDenied);
        assert_eq!(
            fs::read(outside.path().join("output.csv")).unwrap(),
            b"outside"
        );
        assert_eq!(fs::read_dir(outside.path()).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn staged_replacement_replaces_destination_symlink_not_external_target() {
        use std::os::unix::fs::symlink;

        let root = tempdir().expect("root");
        let outside = tempdir().expect("outside");
        let external = outside.path().join("target.csv");
        fs::write(&external, b"external sentinel").expect("outside sentinel");
        symlink(&external, root.path().join("linked.csv")).expect("destination symlink");
        let provider = SandboxFsProvider::new(root.path().to_path_buf()).expect("sandbox");
        let mut stage = provider
            .begin_staged_replace(Path::new("/linked.csv"))
            .expect("stage")
            .expect("sandbox capability");
        executor::block_on(stage.write_chunk(b"replacement")).expect("write");
        executor::block_on(stage.publish()).expect("publish replaces the link entry");
        assert_eq!(
            fs::read(root.path().join("linked.csv")).unwrap(),
            b"replacement"
        );
        assert_eq!(fs::read(external).unwrap(), b"external sentinel");
    }
}
