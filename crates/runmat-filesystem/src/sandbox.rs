#[cfg(not(target_arch = "wasm32"))]
use crate::data_contract::{
    DataChunkUploadRequest, DataChunkUploadTarget, DataManifestDescriptor, DataManifestRequest,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::{DirEntry, FsFileType, FsMetadata, FsProvider, OpenFlags};
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
/// Filesystem provider that sandboxes all operations under a fixed root directory.
///
/// Incoming paths (absolute or relative) are normalized and resolved relative to the sandbox root.
/// Attempts to traverse outside the root using `..` simply clamp to the root, preventing escape.
pub struct SandboxFsProvider {
    root: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl SandboxFsProvider {
    /// Create a new sandbox rooted at `root`. The directory is created if it does not exist.
    pub fn new(root: PathBuf) -> io::Result<Self> {
        if !root.exists() {
            fs::create_dir_all(&root)?;
        }
        let canonical = fs::canonicalize(root)?;
        Ok(Self { root: canonical })
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
        let mut virt = PathBuf::new();
        #[cfg(windows)]
        {
            let prefix = self
                .root
                .components()
                .next()
                .and_then(|component| match component {
                    Component::Prefix(prefix) => Some(prefix.as_os_str()),
                    _ => None,
                });
            if let Some(prefix) = prefix {
                let mut root = OsString::from(prefix);
                root.push(std::path::MAIN_SEPARATOR.to_string());
                virt.push(root);
            } else {
                virt.push(std::path::MAIN_SEPARATOR.to_string());
            }
        }
        #[cfg(not(windows))]
        {
            virt.push(std::path::MAIN_SEPARATOR.to_string());
        }
        if !relative.as_os_str().is_empty() {
            virt.push(relative);
        }
        virt
    }

    fn make_dir_entry(&self, real_path: PathBuf, file_name: OsString) -> DirEntry {
        let file_type = fs::metadata(&real_path)
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
    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn crate::FileHandle>> {
        let target = self.resolve(path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut opts = fs::OpenOptions::new();
        opts.read(flags.read);
        opts.write(flags.write);
        opts.append(flags.append);
        opts.truncate(flags.truncate);
        opts.create(flags.create);
        opts.create_new(flags.create_new);
        let file = opts.open(&target)?;
        Ok(Box::new(file))
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let target = self.resolve(path);
        fs::read(target)
    }

    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        let target = self.resolve(path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(target, data)
    }

    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        let target = self.resolve(path);
        if target.exists() {
            fs::remove_file(target)?;
        }
        Ok(())
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        let target = self.resolve(path);
        fs::metadata(target).map(FsMetadata::from)
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        let target = self.resolve(path);
        fs::symlink_metadata(target).map(FsMetadata::from)
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let target = self.resolve(path);
        let entries = fs::read_dir(&target)?;
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry?;
            out.push(self.make_dir_entry(entry.path(), entry.file_name()));
        }
        Ok(out)
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<PathBuf> {
        let target = self.resolve(path);
        let real = fs::canonicalize(target)?;
        Ok(self.virtualize(&real))
    }

    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        let target = self.resolve(path);
        fs::create_dir(&target)
    }

    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        let target = self.resolve(path);
        fs::create_dir_all(&target)
    }

    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        let target = self.resolve(path);
        fs::remove_dir(&target)
    }

    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        let target = self.resolve(path);
        if target.exists() {
            fs::remove_dir_all(&target)?;
        }
        Ok(())
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        let src = self.resolve(from);
        let dst = self.resolve(to);
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::rename(src, dst)
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        let target = self.resolve(path);
        let mut perms = fs::metadata(&target)?.permissions();
        perms.set_readonly(readonly);
        fs::set_permissions(target, perms)
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
    use super::SandboxFsProvider;
    use crate::FsProvider;
    use futures::executor;
    use std::path::Path;
    use tempfile::tempdir;

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
        assert!(listing
            .iter()
            .any(|entry| entry.path().ends_with(Path::new("nested/sub"))));

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
        assert!(canonical.ends_with(Path::new("data/file.bin")));
        assert!(canonical.is_absolute());
    }
}
