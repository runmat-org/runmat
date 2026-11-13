use crate::{FileHandle, FsFileType, FsMetadata, FsProvider, OpenFlags};
use async_trait::async_trait;
use std::{
    collections::BTreeMap,
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub(super) enum Output {
    Directory,
    File { version: String, size: u64 },
}
pub(super) type OutputRegistry = Arc<Mutex<BTreeMap<PathBuf, Output>>>;

impl Output {
    pub(super) fn version(&self) -> io::Result<&str> {
        match self {
            Self::File { version, .. } => Ok(version),
            Self::Directory => Err(io::Error::other("output is a directory")),
        }
    }
    pub(super) fn metadata(&self) -> FsMetadata {
        match self {
            Self::File { size, .. } => FsMetadata::new(FsFileType::File, *size, None, false),
            Self::Directory => FsMetadata::new(FsFileType::Directory, 0, None, false),
        }
    }
    pub(super) async fn read(&self, base: &dyn FsProvider, path: &Path) -> io::Result<Vec<u8>> {
        let Self::File { version, size } = self else {
            return Err(io::Error::other("output is a directory"));
        };
        let bytes = base.read_version(path, version, *size, "").await?;
        if bytes.len() as u64 != *size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "output version length differs",
            ));
        }
        Ok(bytes)
    }
}

pub(super) struct OutputHandle {
    base: Arc<dyn FsProvider>,
    registry: OutputRegistry,
    path: PathBuf,
    version: String,
    bytes: Cursor<Vec<u8>>,
    append: bool,
    readable: bool,
    writable: bool,
    dirty: bool,
}
impl OutputHandle {
    pub(super) async fn open(
        base: Arc<dyn FsProvider>,
        registry: OutputRegistry,
        path: PathBuf,
        output: Output,
        flags: &OpenFlags,
    ) -> io::Result<Self> {
        let bytes = Cursor::new(output.read(base.as_ref(), &path).await?);
        Ok(Self {
            base,
            registry,
            path,
            version: output.version()?.to_owned(),
            bytes,
            append: flags.append,
            readable: flags.read,
            writable: flags.write || flags.append,
            dirty: false,
        })
    }
}
impl Read for OutputHandle {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if !self.readable {
            return Err(io::ErrorKind::PermissionDenied.into());
        }
        self.bytes.read(buffer)
    }
}
impl Seek for OutputHandle {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.bytes.seek(position)
    }
}
impl Write for OutputHandle {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if !self.writable {
            return Err(io::ErrorKind::PermissionDenied.into());
        }
        if self.append {
            self.bytes.seek(SeekFrom::End(0))?;
        }
        let count = self.bytes.write(bytes)?;
        self.dirty |= count > 0;
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.dirty {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "conditional output handles require asynchronous flush",
            ));
        }
        Ok(())
    }
}
#[async_trait(?Send)]
impl FileHandle for OutputHandle {
    async fn flush_async(&mut self) -> io::Result<()> {
        if !self.dirty {
            return Ok(());
        }
        let bytes = self.bytes.get_ref();
        let version = self
            .base
            .publish_output(&self.path, bytes, Some(&self.version))
            .await?;
        self.registry.lock().expect("output ownership lock").insert(
            self.path.clone(),
            Output::File {
                version: version.clone(),
                size: bytes.len() as u64,
            },
        );
        self.version = version;
        self.dirty = false;
        Ok(())
    }
    async fn metadata_async(&self) -> io::Result<FsMetadata> {
        Ok(FsMetadata::new(
            FsFileType::File,
            self.bytes.get_ref().len() as u64,
            None,
            false,
        ))
    }
}
