use crate::data_contract::{
    DataChunkUploadRequest, DataChunkUploadTarget, DataManifestDescriptor, DataManifestRequest,
};
use crate::{DirEntry, FsMetadata, FsProvider, OpenFlags};
use async_trait::async_trait;
use std::io::{self, ErrorKind};
use std::path::Path;

#[derive(Default)]
pub struct PlaceholderFsProvider;

#[async_trait(?Send)]
impl FsProvider for PlaceholderFsProvider {
    fn open(&self, _path: &Path, _flags: &OpenFlags) -> io::Result<Box<dyn crate::FileHandle>> {
        Err(unsupported())
    }

    async fn read(&self, _path: &Path) -> io::Result<Vec<u8>> {
        Err(unsupported())
    }

    async fn write(&self, _path: &Path, _data: &[u8]) -> io::Result<()> {
        Err(unsupported())
    }

    async fn remove_file(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    async fn metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
        Err(unsupported())
    }

    async fn symlink_metadata(&self, _path: &Path) -> io::Result<FsMetadata> {
        Err(unsupported())
    }

    async fn read_dir(&self, _path: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unsupported())
    }

    async fn canonicalize(&self, _path: &Path) -> io::Result<std::path::PathBuf> {
        Err(unsupported())
    }

    async fn create_dir(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    async fn create_dir_all(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    async fn remove_dir(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    async fn remove_dir_all(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    async fn rename(&self, _from: &Path, _to: &Path) -> io::Result<()> {
        Err(unsupported())
    }

    async fn set_readonly(&self, _path: &Path, _readonly: bool) -> io::Result<()> {
        Err(unsupported())
    }

    async fn data_manifest_descriptor(
        &self,
        _request: &DataManifestRequest,
    ) -> io::Result<DataManifestDescriptor> {
        Err(unsupported())
    }

    async fn data_chunk_upload_targets(
        &self,
        _request: &DataChunkUploadRequest,
    ) -> io::Result<Vec<DataChunkUploadTarget>> {
        Err(unsupported())
    }

    async fn data_upload_chunk(
        &self,
        _target: &DataChunkUploadTarget,
        _data: &[u8],
    ) -> io::Result<()> {
        Err(unsupported())
    }
}

fn unsupported() -> io::Error {
    io::Error::new(
        ErrorKind::Unsupported,
        "filesystem provider not installed for wasm target",
    )
}
