use crate::data_contract::{
    DataChunkUploadRequest, DataChunkUploadTarget, DataManifestDescriptor, DataManifestRequest,
};
use crate::{
    DirEntry, FileHandle, FsBoundedInput, FsFilePayloadContract, FsFileType,
    FsManagedOpenedInputCost, FsManagedReadBacking, FsManagedReadContract, FsMetadata, FsProvider,
    FsResourceContract, FsResourceLease, FsResourceLeaseHandle, FsResourceRevision, OpenFlags,
};
use async_trait::async_trait;
use chrono::Utc;
use serde_json::Value as JsonValue;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use url::Url;

#[path = "native/staged.rs"]
pub(crate) mod staged;
#[path = "native/mapped.rs"]
mod mapped;

#[derive(Default)]
pub struct NativeFsProvider;

#[async_trait(?Send)]
impl FsProvider for NativeFsProvider {
    fn path_namespace(&self) -> crate::FsPathNamespace {
        crate::FsPathNamespace::NativeOs
    }

    async fn open_mapped_backing(
        &self,
        path: &Path,
        writable: bool,
    ) -> io::Result<Option<Box<dyn crate::FsMappedBacking>>> {
        mapped::open(path, writable).map(Some)
    }

    fn begin_staged_replace(
        &self,
        destination: &Path,
    ) -> io::Result<Option<Box<dyn crate::FsStagedOutput>>> {
        staged::begin(destination).map(Some)
    }

    fn acquire_native_path(
        &self,
        path: &Path,
        intent: crate::NativePathIntent,
    ) -> io::Result<Option<crate::NativePathGrant>> {
        if intent == crate::NativePathIntent::ProcessWorkingDirectory && !path.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "process working directory is not an existing native directory",
            ));
        }
        struct StableNativeMapping;
        impl crate::NativePathAuthority for StableNativeMapping {
            fn is_valid(&self) -> bool {
                true
            }

            fn process_revocation(&self) -> crate::NativeProcessRevocation<'_> {
                crate::NativeProcessRevocation::Stable
            }
        }
        crate::NativePathGrant::issued_by_provider(
            path.to_path_buf(),
            intent,
            Box::new(StableNativeMapping),
        )
        .map(Some)
    }

    async fn open_bounded_input(&self, path: &Path) -> io::Result<Option<FsBoundedInput>> {
        if !path.is_absolute() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "bounded native input requires a resolved absolute path",
            ));
        }
        let file = fs::File::open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Ok(None);
        }
        let metadata = FsMetadata::from(metadata);
        FsBoundedInput::new_with_lease(
            Box::new(file),
            metadata,
            FsResourceLeaseHandle::Static(&NATIVE_DIRECT_LEASE),
        )
        .map(Some)
    }

    fn managed_read_contract(&self) -> Option<FsManagedReadContract> {
        Some(FsManagedReadContract {
            revision: native_direct_revision(),
            backing: FsManagedReadBacking::ExternalToRustManagedHeap,
            capture_requested_heap_bytes: 0,
            opened_requested_heap_bytes: std::mem::size_of::<fs::File>() as u64,
        })
    }

    fn resource_lease(&self) -> Option<Arc<dyn FsResourceLease>> {
        Some(Arc::new(NativeDirectFileLease))
    }

    fn resource_lease_handle(&self) -> Option<FsResourceLeaseHandle> {
        Some(FsResourceLeaseHandle::Static(&NATIVE_DIRECT_LEASE))
    }

    fn open(&self, path: &Path, flags: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        let mut opts = fs::OpenOptions::new();
        opts.read(flags.read);
        opts.write(flags.write);
        opts.append(flags.append);
        opts.truncate(flags.truncate);
        opts.create(flags.create);
        opts.create_new(flags.create_new);
        let file = opts.open(path)?;
        Ok(Box::new(file))
    }

    async fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        fs::read(path)
    }

    async fn write(&self, path: &Path, data: &[u8]) -> io::Result<()> {
        fs::write(path, data)
    }

    async fn remove_file(&self, path: &Path) -> io::Result<()> {
        fs::remove_file(path)
    }

    async fn metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        fs::metadata(path).map(FsMetadata::from)
    }

    async fn symlink_metadata(&self, path: &Path) -> io::Result<FsMetadata> {
        fs::symlink_metadata(path).map(FsMetadata::from)
    }

    async fn read_dir(&self, path: &Path) -> io::Result<Vec<DirEntry>> {
        let entries = fs::read_dir(path)?;
        let mut out = Vec::new();
        for entry in entries {
            let entry = entry?;
            let file_type = entry
                .file_type()
                .ok()
                .map(FsFileType::from)
                .unwrap_or(FsFileType::Unknown);
            out.push(DirEntry {
                path: entry.path(),
                file_name: entry.file_name(),
                file_type,
            });
        }
        Ok(out)
    }

    async fn canonicalize(&self, path: &Path) -> io::Result<std::path::PathBuf> {
        fs::canonicalize(path)
    }

    async fn create_dir(&self, path: &Path) -> io::Result<()> {
        fs::create_dir(path)
    }

    async fn create_dir_all(&self, path: &Path) -> io::Result<()> {
        fs::create_dir_all(path)
    }

    async fn remove_dir(&self, path: &Path) -> io::Result<()> {
        fs::remove_dir(path)
    }

    async fn remove_dir_all(&self, path: &Path) -> io::Result<()> {
        fs::remove_dir_all(path)
    }

    async fn rename(&self, from: &Path, to: &Path) -> io::Result<()> {
        fs::rename(from, to)
    }

    async fn rename_replace(&self, from: &Path, to: &Path) -> io::Result<()> {
        fs::rename(from, to)
    }

    async fn set_readonly(&self, path: &Path, readonly: bool) -> io::Result<()> {
        let mut perms = fs::metadata(path)?.permissions();
        perms.set_readonly(readonly);
        fs::set_permissions(path, perms)
    }

    async fn data_manifest_descriptor(
        &self,
        request: &DataManifestRequest,
    ) -> io::Result<DataManifestDescriptor> {
        let path = dataset_manifest_path(&request.path);
        let bytes = fs::read(&path)?;
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
        let root = dataset_chunk_root(&request.dataset_path, &request.array);
        fs::create_dir_all(&root)?;
        request
            .chunks
            .iter()
            .map(|chunk| {
                let path = root.join(format!("{}.bin", sanitize_segment(&chunk.object_id)));
                let canonical = path_to_file_url(&path)?;
                Ok(DataChunkUploadTarget {
                    key: chunk.key.clone(),
                    method: "PUT".to_string(),
                    upload_url: canonical,
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
        let path = file_url_to_path(&target.upload_url)?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, data)
    }
}

struct NativeDirectFileLease;
static NATIVE_DIRECT_LEASE: NativeDirectFileLease = NativeDirectFileLease;

fn native_direct_revision() -> FsResourceRevision {
    FsResourceRevision::for_provider(
        "runmat.filesystem.native",
        b"direct-file-handles-bounded-open-managed-v3",
    )
}

impl FsResourceLease for NativeDirectFileLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: native_direct_revision(),
            opened_input_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
            opened_output_handle: Some(FsFilePayloadContract::CallerOwnedBuffer),
        }
    }

    fn is_valid(&self) -> bool {
        true // NativeFsProvider has no mutable configuration.
    }

    fn managed_opened_input_cost(&self) -> Option<FsManagedOpenedInputCost> {
        // Only the Box<File> is freshly requested. The immutable lease is
        // static; OS descriptor/kernel/stdlib internals remain out of scope.
        Some(FsManagedOpenedInputCost {
            requested_heap_bytes: std::mem::size_of::<fs::File>() as u64,
        })
    }
}

fn dataset_manifest_path(path: &str) -> PathBuf {
    if path.ends_with(".json") {
        return PathBuf::from(path);
    }
    PathBuf::from(path).join("manifest.json")
}

fn dataset_chunk_root(dataset_path: &str, array: &str) -> PathBuf {
    PathBuf::from(dataset_path)
        .join("arrays")
        .join(sanitize_segment(array))
        .join("chunks")
}

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

fn path_to_file_url(path: &Path) -> io::Result<String> {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    Url::from_file_path(abs)
        .map(|url| url.to_string())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid local path"))
}

fn file_url_to_path(upload_url: &str) -> io::Result<PathBuf> {
    let url = Url::parse(upload_url)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidInput, err.to_string()))?;
    if url.scheme() != "file" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unsupported upload url scheme '{}'", url.scheme()),
        ));
    }
    url.to_file_path().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "failed to decode local file upload url",
        )
    })
}

impl From<fs::Metadata> for FsMetadata {
    fn from(meta: fs::Metadata) -> Self {
        let file_type = FsFileType::from(meta.file_type());
        FsMetadata {
            file_type,
            len: meta.len(),
            modified: meta.modified().ok(),
            readonly: meta.permissions().readonly(),
            hash: None,
        }
    }
}

impl From<fs::FileType> for FsFileType {
    fn from(ft: fs::FileType) -> Self {
        if ft.is_dir() {
            FsFileType::Directory
        } else if ft.is_file() {
            FsFileType::File
        } else if ft.is_symlink() {
            FsFileType::Symlink
        } else {
            FsFileType::Other
        }
    }
}
