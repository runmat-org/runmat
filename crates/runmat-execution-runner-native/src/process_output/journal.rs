use std::fs::File;
#[cfg(unix)]
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use runmat_execution::{identity::AttemptId, Digest};
use serde::{Deserialize, Serialize};

use super::budget::METADATA_FILE_BYTES;
use super::{OutputResult, RawOutputBudget, RawOutputError, RawOutputStream, OUTPUT_PAGE_BYTES};
use crate::supervisor::filesystem::{
    atomic_json, immutable_json, read_optional_json_bounded, secure_directory,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawOutputIdentity {
    pub(crate) attempt: AttemptId,
    pub(crate) request: Digest,
}

impl RawOutputIdentity {
    pub(crate) fn for_attempt(
        request: &runmat_execution_runner::AttemptRequest,
    ) -> OutputResult<Self> {
        Ok(Self {
            attempt: request.id,
            request: crate::metadata_identity::json_digest(
                b"runmat-original-raw-process-output-v1\0",
                request,
                u64::from(crate::supervisor::protocol::SUPERVISOR_MAX_MESSAGE_BYTES),
            )?,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RawOutputState {
    Reading,
    EndOfStream,
    ReadFailed,
    CaptureFailed,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    schema_version: u16,
    identity: RawOutputIdentity,
    budget: RawOutputBudget,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamHead {
    bytes: u64,
    state: RawOutputState,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Head {
    identity: RawOutputIdentity,
    streams: [StreamHead; 2],
    // This is set by the process owner, never by a pipe or producer receipt.
    retired: bool,
}

pub(crate) struct RawOutputJournal {
    reader: RawOutputReader,
    files: [File; 2],
    head: Head,
    failed: bool,
}

pub(crate) struct RawOutputReader {
    root: ReaderRoot,
    header: Header,
}

enum ReaderRoot {
    Process(PathBuf),
    Claimed(runmat_runtime::context::host_allocation::HostControlVec<u8>),
}

impl std::ops::Deref for ReaderRoot {
    type Target = Path;
    fn deref(&self) -> &Path {
        match self {
            Self::Process(path) => path,
            Self::Claimed(bytes) => {
                // SAFETY: these unchanged bytes were copied from a complete
                // OsStr::as_encoded_bytes in this process/Rust target. They
                // are never accepted from persistence or a transport.
                Path::new(unsafe { std::ffi::OsStr::from_encoded_bytes_unchecked(bytes) })
            }
        }
    }
}

impl RawOutputJournal {
    pub(crate) fn for_attempt(
        root: PathBuf,
        attempt: &runmat_execution_runner::AttemptRequest,
        materialized: u64,
    ) -> OutputResult<Self> {
        attempt
            .validate()
            .map_err(crate::NativeExecutionError::from)?;
        let budget = RawOutputBudget::from_grant(
            attempt.task.resources.scratch_bytes,
            materialized,
            attempt.task.console,
        )?;
        Self::create(root, RawOutputIdentity::for_attempt(attempt)?, budget)
    }

    pub(crate) fn create(
        root: PathBuf,
        identity: RawOutputIdentity,
        budget: RawOutputBudget,
    ) -> OutputResult<Self> {
        secure_directory(&root)?;
        let stdout = create_stream(&root, RawOutputStream::Stdout)?;
        let stderr = create_stream(&root, RawOutputStream::Stderr)?;
        let header = Header {
            schema_version: 1,
            identity: identity.clone(),
            budget,
        };
        let head = Head {
            identity,
            streams: [StreamHead {
                bytes: 0,
                state: RawOutputState::Reading,
            }; 2],
            retired: false,
        };
        // Publish discovery last. A failed initialization never exposes a
        // header whose streams or initial head were not created successfully.
        immutable_json(&root.join("head.json"), &head)?;
        immutable_json(&root.join("original.json"), &header)?;
        Ok(Self {
            reader: RawOutputReader {
                root: ReaderRoot::Process(root),
                header,
            },
            files: [stdout, stderr],
            head,
            failed: false,
        })
    }

    pub(crate) fn reader_ref(&self) -> &RawOutputReader {
        &self.reader
    }

    #[cfg(test)]
    pub(crate) fn reader(&self) -> RawOutputReader {
        RawOutputReader {
            root: ReaderRoot::Process(self.reader.root.to_path_buf()),
            header: self.reader.header.clone(),
        }
    }

    pub(crate) fn append(&mut self, stream: RawOutputStream, bytes: &[u8]) -> OutputResult<()> {
        if self.failed
            || self.head.retired
            || self.head.streams[stream.index()].state != RawOutputState::Reading
        {
            return Err(RawOutputError::Invalid(
                "journal or stream is no longer writable",
            ));
        }
        let result = self.append_open(stream, bytes);
        self.failed |= result.is_err();
        result
    }

    fn append_open(&mut self, stream: RawOutputStream, bytes: &[u8]) -> OutputResult<()> {
        let required = self
            .head
            .streams
            .iter()
            .try_fold(bytes.len() as u64, |sum, head| sum.checked_add(head.bytes))
            .ok_or(RawOutputError::Invalid("stream byte count overflows"))?;
        let maximum = self.reader.header.budget.payload_bytes();
        if required > maximum {
            return Err(RawOutputError::StorageLimit { required, maximum });
        }
        if bytes.is_empty() {
            return Ok(());
        }
        let file = &mut self.files[stream.index()];
        file.write_all(bytes)?;
        file.sync_data()?;
        let mut next = self.head.clone();
        next.streams[stream.index()].bytes += bytes.len() as u64;
        self.publish_head(next)
    }

    pub(crate) fn finish_stream(
        &mut self,
        stream: RawOutputStream,
        state: RawOutputState,
    ) -> OutputResult<()> {
        if state == RawOutputState::Reading
            || self.head.streams[stream.index()].state != RawOutputState::Reading
        {
            return Err(RawOutputError::Invalid(
                "stream completion is not a single terminal transition",
            ));
        }
        let mut next = self.head.clone();
        next.streams[stream.index()].state = if self.failed {
            RawOutputState::CaptureFailed
        } else {
            state
        };
        self.publish_head(next)
    }

    pub(crate) fn mark_retired(&mut self) -> OutputResult<()> {
        if self
            .head
            .streams
            .iter()
            .any(|stream| stream.state == RawOutputState::Reading)
        {
            return Err(RawOutputError::Invalid(
                "retirement publication requires both stream outcomes",
            ));
        }
        let mut next = self.head.clone();
        next.retired = true;
        self.publish_head(next)
    }

    /// After the pipe owner stops polling, close only streams that still lack
    /// an outcome. Preserve already observed EOF and every committed byte.
    pub(crate) fn fail_open_streams(&mut self) -> OutputResult<()> {
        if self
            .head
            .streams
            .iter()
            .all(|stream| stream.state != RawOutputState::Reading)
        {
            return Ok(());
        }
        let mut next = self.head.clone();
        for stream in &mut next.streams {
            if stream.state == RawOutputState::Reading {
                stream.state = if self.failed {
                    RawOutputState::CaptureFailed
                } else {
                    RawOutputState::ReadFailed
                };
            }
        }
        self.publish_head(next)
    }

    fn publish_head(&mut self, next: Head) -> OutputResult<()> {
        // Atomic publication may become visible before retention fails. Keep
        // the new monotone inventory even on error, so a later terminal write
        // cannot replace a visible head with an older byte count. The original
        // error still fails the attempt; reaffirmation is not a successful run.
        self.head = next;
        let result = atomic_json(&self.reader.root.join("head.json"), &self.head)
            .map_err(RawOutputError::from);
        self.failed |= result.is_err();
        result
    }
}

impl RawOutputReader {
    pub(crate) fn open(
        root: PathBuf,
        identity: RawOutputIdentity,
        budget: RawOutputBudget,
    ) -> OutputResult<Self> {
        let header: Header = read_required(&root.join("original.json"))?;
        if header.schema_version != 1
            || header.identity != identity
            || header.budget.payload_bytes() > budget.payload_bytes()
        {
            return Err(RawOutputError::Invalid(
                "journal differs from the exact original attempt or storage grant",
            ));
        }
        Ok(Self {
            root: ReaderRoot::Process(root),
            header,
        })
    }

    pub(super) fn retained_copy(
        &self,
        owner: Option<&runmat_runtime::context::host_allocation::HostControlAllocationOwner>,
    ) -> OutputResult<Self> {
        let source = self.root.as_os_str().as_encoded_bytes();
        let mut bytes = runmat_runtime::context::host_allocation::HostControlVec::with_capacity(
            source.len(),
            owner,
        )?;
        for byte in source {
            bytes
                .try_push(*byte)
                .map_err(|_| RawOutputError::Invalid("retained path capacity changed"))?;
        }
        Ok(Self {
            root: ReaderRoot::Claimed(bytes),
            header: self.header.clone(),
        })
    }

    /// Read only committed bytes into the caller's already admitted buffer.
    /// Returning EOF here concerns one stream, not physical process retirement.
    pub(crate) fn read_into(
        &self,
        stream: RawOutputStream,
        offset: u64,
        buffer: &mut [u8],
    ) -> OutputResult<(usize, RawOutputState, bool)> {
        if buffer.is_empty() || buffer.len() > OUTPUT_PAGE_BYTES {
            return Err(RawOutputError::Invalid(
                "page buffer is empty or exceeds its bound",
            ));
        }
        let head: Head = read_required(&self.root.join("head.json"))?;
        let total = head
            .streams
            .iter()
            .try_fold(0u64, |sum, stream| sum.checked_add(stream.bytes))
            .ok_or(RawOutputError::Invalid("journal byte count overflows"))?;
        if head.identity != self.header.identity || total > self.header.budget.payload_bytes() {
            return Err(RawOutputError::Invalid(
                "journal head differs from its original authority",
            ));
        }
        let current = head.streams[stream.index()];
        let available = current
            .bytes
            .checked_sub(offset)
            .ok_or(RawOutputError::Invalid(
                "cursor exceeds committed stream bytes",
            ))?;
        let count = available.min(buffer.len() as u64) as usize;
        if count != 0 {
            let mut file = open_stream(&self.root.join(stream.file_name()))?;
            file.seek(SeekFrom::Start(offset))?;
            file.read_exact(&mut buffer[..count])?;
        }
        let state = if available > count as u64 {
            RawOutputState::Reading
        } else {
            current.state
        };
        Ok((count, state, head.retired))
    }
}

#[cfg(unix)]
fn create_stream(root: &Path, stream: RawOutputStream) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    use std::os::unix::fs::OpenOptionsExt as _;
    options
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    let file = options.open(root.join(stream.file_name()))?;
    file.sync_all()?;
    Ok(file)
}

#[cfg(windows)]
fn create_stream(root: &Path, stream: RawOutputStream) -> std::io::Result<File> {
    let file = runmat_filesystem::NativeFsProvider
        .create_owner_private_file_new(&root.join(stream.file_name()))?;
    file.sync_all()?;
    Ok(file)
}

#[cfg(windows)]
fn open_stream(path: &Path) -> std::io::Result<File> {
    runmat_filesystem::NativeFsProvider.open_owner_private_file(path)
}

#[cfg(unix)]
fn open_stream(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::{MetadataExt as _, OpenOptionsExt as _};
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.mode() & 0o077 != 0
        || metadata.uid() != unsafe { libc::geteuid() }
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "raw output file is not owner-private",
        ));
    }
    Ok(file)
}

fn read_required<T: serde::de::DeserializeOwned>(path: &Path) -> OutputResult<T> {
    read_optional_json_bounded(path, METADATA_FILE_BYTES)?.ok_or(RawOutputError::Invalid(
        "required journal metadata is absent",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(seed: &[u8]) -> RawOutputIdentity {
        RawOutputIdentity {
            attempt: AttemptId::derive(&[seed]),
            request: Digest::sha256(seed),
        }
    }

    #[test]
    fn committed_raw_pages_bind_attempt_and_distinguish_eof_from_retirement() {
        let temp = tempfile::tempdir().unwrap();
        #[cfg(target_os = "linux")]
        let leaf = {
            use std::os::unix::ffi::OsStringExt;
            let mut bytes = "output-λ".as_bytes().to_vec();
            bytes.push(0xff);
            std::ffi::OsString::from_vec(bytes)
        };
        #[cfg(not(target_os = "linux"))]
        let leaf = std::ffi::OsString::from("output-λ");
        let root = temp.path().join(leaf);
        let budget = RawOutputBudget::from_grant(METADATA_FILE_BYTES * 3 + 64, 0, None).unwrap();
        let mut journal = RawOutputJournal::create(root.clone(), identity(b"one"), budget).unwrap();
        journal
            .append(RawOutputStream::Stdout, &[0, 255, 1, 2])
            .unwrap();
        journal.append(RawOutputStream::Stderr, b"error").unwrap();
        let reader = RawOutputReader::open(root.clone(), identity(b"one"), budget)
            .unwrap()
            .retained_copy(None)
            .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            // Retention must preserve native path bytes even on filesystems
            // that reject non-UTF-8 names (including macOS). Test that copy
            // independently of creating such a directory on the host volume.
            let encoded_root = root.join(std::ffi::OsString::from_vec(vec![0xff]));
            let original = RawOutputReader {
                root: ReaderRoot::Process(encoded_root.clone()),
                header: reader.header.clone(),
            };
            let retained = original.retained_copy(None).unwrap();
            assert_eq!(retained.root.as_os_str(), encoded_root.as_os_str());
        }
        assert!(RawOutputReader::open(root, identity(b"other"), budget).is_err());
        let mut buffer = [0; 3];
        assert_eq!(
            reader
                .read_into(RawOutputStream::Stdout, 0, &mut buffer)
                .unwrap(),
            (3, RawOutputState::Reading, false)
        );
        assert_eq!(buffer, [0, 255, 1]);
        journal
            .finish_stream(RawOutputStream::Stdout, RawOutputState::EndOfStream)
            .unwrap();
        journal
            .finish_stream(RawOutputStream::Stderr, RawOutputState::EndOfStream)
            .unwrap();
        assert_eq!(
            reader
                .read_into(RawOutputStream::Stdout, 3, &mut buffer)
                .unwrap(),
            (1, RawOutputState::EndOfStream, false)
        );
        assert_eq!(buffer[0], 2);
        journal.mark_retired().unwrap();
        assert_eq!(
            reader
                .read_into(RawOutputStream::Stdout, 4, &mut buffer)
                .unwrap(),
            (0, RawOutputState::EndOfStream, true)
        );
        assert!(reader
            .read_into(RawOutputStream::Stdout, 5, &mut buffer)
            .is_err());
    }

    #[test]
    fn both_streams_share_one_grant_and_failed_append_is_not_published() {
        let temp = tempfile::tempdir().unwrap();
        let budget = RawOutputBudget::from_grant(METADATA_FILE_BYTES * 3 + 5, 0, None).unwrap();
        let mut journal =
            RawOutputJournal::create(temp.path().join("output"), identity(b"bounded"), budget)
                .unwrap();
        let reader = journal.reader();
        journal.append(RawOutputStream::Stdout, b"abc").unwrap();
        assert!(matches!(
            journal.append(RawOutputStream::Stderr, b"def"),
            Err(RawOutputError::StorageLimit {
                required: 6,
                maximum: 5
            })
        ));
        assert!(journal.append(RawOutputStream::Stdout, b"d").is_err());
        journal
            .finish_stream(RawOutputStream::Stdout, RawOutputState::EndOfStream)
            .unwrap();
        journal
            .finish_stream(RawOutputStream::Stderr, RawOutputState::EndOfStream)
            .unwrap();
        let mut buffer = [0; 8];
        assert_eq!(
            reader
                .read_into(RawOutputStream::Stderr, 0, &mut buffer)
                .unwrap(),
            (0, RawOutputState::CaptureFailed, false)
        );
        assert_eq!(
            reader
                .read_into(RawOutputStream::Stdout, 0, &mut buffer)
                .unwrap(),
            (3, RawOutputState::CaptureFailed, false)
        );
        assert_eq!(&buffer[..3], b"abc");
    }

    #[test]
    fn interrupted_drain_preserves_completed_stream_and_published_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let budget = RawOutputBudget::from_grant(METADATA_FILE_BYTES * 3 + 5, 0, None).unwrap();
        let mut journal =
            RawOutputJournal::create(temp.path().join("output"), identity(b"interrupted"), budget)
                .unwrap();
        journal.append(RawOutputStream::Stdout, b"ok").unwrap();
        journal
            .finish_stream(RawOutputStream::Stdout, RawOutputState::EndOfStream)
            .unwrap();
        journal.append(RawOutputStream::Stderr, b"err").unwrap();
        journal.fail_open_streams().unwrap();
        journal.fail_open_streams().unwrap();
        let mut buffer = [0; 8];
        let reader = journal.reader();
        assert_eq!(
            reader
                .read_into(RawOutputStream::Stdout, 0, &mut buffer)
                .unwrap(),
            (2, RawOutputState::EndOfStream, false)
        );
        assert_eq!(&buffer[..2], b"ok");
        assert_eq!(
            reader
                .read_into(RawOutputStream::Stderr, 0, &mut buffer)
                .unwrap(),
            (3, RawOutputState::ReadFailed, false)
        );
        assert_eq!(&buffer[..3], b"err");
    }
}
