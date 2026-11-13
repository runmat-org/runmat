use std::fs;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use runmat_execution::identity::ArtifactId;
use runmat_execution::{Digest, DigestStream};
use runmat_filesystem::{AtomicCreateResult, NativeFsProvider};
use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::{NativeExecutionError, NativeExecutionResult};

pub(crate) struct ArtifactStore {
    root: PathBuf,
}

/// Identity of one bounded, exact JSON Graph object in the local store.
/// The child receives only these scalars, never a copy of the request bytes.
pub(crate) struct StreamedGraphArtifact {
    pub(crate) id: ArtifactId,
    pub(crate) digest: Digest,
    pub(crate) length: u64,
}

impl ArtifactStore {
    pub(crate) fn new(root: PathBuf) -> NativeExecutionResult<Self> {
        NativeFsProvider
            .create_private_directory_durable(&root)
            .map_err(io_error)?;
        Ok(Self { root })
    }

    pub(crate) fn put(&self, id: ArtifactId, bytes: &[u8]) -> NativeExecutionResult<()> {
        let path = self.root.join(id.to_string());
        // Publish one complete winner. The shared filesystem owns no-replace
        // linearization; checking existence before an ordinary write does not.
        let publication = NativeFsProvider
            .atomic_create_new_sync(&path, bytes)
            .map_err(io_error)?;
        if let AtomicCreateResult::Existing(existing) = publication {
            if existing != bytes {
                return Err(NativeExecutionError::Protocol(
                    "local artifact store identity collision".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn get(&self, id: ArtifactId, maximum_bytes: u64) -> NativeExecutionResult<Vec<u8>> {
        NativeFsProvider
            .read_bounded_sync(&self.root.join(id.to_string()), maximum_bytes)
            .map_err(io_error)
    }

    /// Serialize once into an unpublished private file, hashing and bounding
    /// the exact bytes that will be opened by the worker. Publication is
    /// create-new; an existing identity must contain those exact bytes.
    pub(crate) fn put_streamed_graph<T: Serialize>(
        &self,
        value: &T,
        maximum_length: u64,
    ) -> NativeExecutionResult<StreamedGraphArtifact> {
        let mut temporary = tempfile::NamedTempFile::new_in(&self.root).map_err(io_error)?;
        let (digest, length) = {
            // Coalesce the serializer's tiny numeric/token writes before both
            // hashing and file I/O. Keep one bounded-size buffer, as before.
            let mut writer = BufWriter::new(BoundedHashWriter {
                inner: temporary.as_file_mut(),
                digest: DigestStream::new(),
                length: 0,
                maximum: maximum_length,
            });
            serde_json::to_writer(&mut writer, value)
                .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
            writer.flush().map_err(io_error)?;
            let writer = writer
                .into_inner()
                .map_err(|error| io_error(error.into_error()))?;
            (writer.digest.finish(), writer.length)
        };
        if length == 0 {
            return Err(NativeExecutionError::Protocol(
                "stored Graph request is empty".into(),
            ));
        }
        let id = ArtifactId::derive(&[b"graph-worker-request-v10", digest.bytes()]);
        let path = self.root.join(id.to_string());
        match temporary.persist_noclobber(&path) {
            Ok(_) => {}
            Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
                let (existing_digest, existing_length) = hash_file_bounded(&path, maximum_length)?;
                if existing_digest != digest || existing_length != length {
                    return Err(NativeExecutionError::Protocol(
                        "local Graph artifact identity collision".into(),
                    ));
                }
            }
            Err(error) => return Err(io_error(error.error)),
        }
        Ok(StreamedGraphArtifact { id, digest, length })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    /// A child may reopen only the parent's typed artifact-store root. The
    /// expected digest/length are carried in its bounded launch descriptor;
    /// a hashing reader authenticates the very bytes used by deserialization.
    pub(crate) fn from_existing_root(root: PathBuf) -> NativeExecutionResult<Self> {
        if !root.is_dir() {
            return Err(NativeExecutionError::Protocol(
                "native artifact-store root is unavailable".into(),
            ));
        }
        Ok(Self { root })
    }

    pub(crate) fn read_verified<T: DeserializeOwned>(
        &self,
        id: ArtifactId,
        expected_digest: &Digest,
        expected_length: u64,
        maximum_length: u64,
    ) -> NativeExecutionResult<T> {
        read_verified_json_file(
            &self.root.join(id.to_string()),
            expected_digest,
            expected_length,
            maximum_length,
        )
    }
}

/// Authenticate the same bounded file bytes consumed by JSON decoding. This
/// is shared by local stdio and signed remote Program staging, with no
/// hash-then-reopen interval or duplicate whole-request buffer.
pub(crate) fn read_verified_json_file<T: DeserializeOwned>(
    path: &Path,
    expected_digest: &Digest,
    expected_length: u64,
    maximum_length: u64,
) -> NativeExecutionResult<T> {
    if expected_length == 0 || expected_length > maximum_length {
        return Err(NativeExecutionError::Protocol(
            "staged Graph artifact length exceeds its admitted bound".into(),
        ));
    }
    let file = fs::File::open(path).map_err(io_error)?;
    // Hash physical read batches, not each byte requested by the JSON parser.
    // The same single buffer still bounds read-ahead and no bytes are reopened.
    let mut reader = BufReader::new(DigestingReader {
        inner: file,
        digest: DigestStream::new(),
        length: 0,
        maximum: expected_length,
    });
    let value = serde_json::from_reader(&mut reader)
        .map_err(|error| NativeExecutionError::Protocol(error.to_string()))?;
    let reader = reader.into_inner();
    if reader.length != expected_length || &reader.digest.finish() != expected_digest {
        return Err(NativeExecutionError::Protocol(
            "staged Graph artifact differs from its launch identity".into(),
        ));
    }
    Ok(value)
}

struct BoundedHashWriter<W> {
    inner: W,
    digest: DigestStream,
    length: u64,
    maximum: u64,
}

impl<W: Write> Write for BoundedHashWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let length = self
            .length
            .checked_add(bytes.len() as u64)
            .filter(|length| *length <= self.maximum)
            .ok_or_else(|| std::io::Error::other("stored Graph request exceeds its wire bound"))?;
        self.inner.write_all(bytes)?;
        self.digest.update(bytes);
        self.length = length;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn hash_file_bounded(path: &Path, maximum_length: u64) -> NativeExecutionResult<(Digest, u64)> {
    let mut reader = BufReader::new(fs::File::open(path).map_err(io_error)?);
    let mut bytes = [0_u8; 64 * 1024];
    let mut digest = DigestStream::new();
    let mut length = 0_u64;
    loop {
        let count = reader.read(&mut bytes).map_err(io_error)?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(count as u64)
            .filter(|length| *length <= maximum_length)
            .ok_or_else(|| {
                NativeExecutionError::Protocol(
                    "stored Graph artifact exceeds its wire bound".into(),
                )
            })?;
        digest.update(&bytes[..count]);
    }
    Ok((digest.finish(), length))
}

struct DigestingReader<R> {
    inner: R,
    digest: DigestStream,
    length: u64,
    maximum: u64,
}

impl<R: Read> Read for DigestingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(buf)?;
        self.length = self
            .length
            .checked_add(count as u64)
            .filter(|length| *length <= self.maximum)
            .ok_or_else(|| std::io::Error::other("staged Graph artifact is oversized"))?;
        self.digest.update(&buf[..count]);
        Ok(count)
    }
}

fn io_error(error: std::io::Error) -> NativeExecutionError {
    NativeExecutionError::Protocol(format!("local artifact store failed: {error}"))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod streamed_graph_tests {
    use serde::{Deserialize, Serialize};

    use super::ArtifactStore;

    #[derive(Debug, Eq, PartialEq, Serialize, Deserialize)]
    struct GraphBytes {
        bytes: Vec<u8>,
    }

    #[test]
    fn streamed_graph_is_one_exact_verified_file() {
        let root = tempfile::tempdir().expect("artifact root");
        let store = ArtifactStore::new(root.path().join("objects")).expect("store");
        let graph = GraphBytes {
            bytes: vec![7; 256 * 1024],
        };
        let first = store
            .put_streamed_graph(&graph, 2 * 1024 * 1024)
            .expect("streamed object");
        let encoded = serde_json::to_vec(&graph).unwrap();
        assert_eq!(first.digest, runmat_execution::Digest::sha256(&encoded));
        assert_eq!(first.length, encoded.len() as u64);
        assert_eq!(store.get(first.id, first.length).unwrap(), encoded);
        assert!(store.put_streamed_graph(&graph, first.length - 1).is_err());
        assert!(store.put_streamed_graph(&graph, first.length).is_ok());
        let second = store
            .put_streamed_graph(&graph, 2 * 1024 * 1024)
            .expect("identical object");
        assert_eq!(first.id, second.id);
        assert_eq!(first.digest, second.digest);
        assert_eq!(first.length, second.length);
        assert_eq!(
            store
                .read_verified::<GraphBytes>(
                    first.id,
                    &first.digest,
                    first.length,
                    2 * 1024 * 1024,
                )
                .expect("one-pass verified read"),
            graph
        );
        assert!(store.put_streamed_graph(&graph, 1024).is_err());
        std::fs::write(store.root().join(first.id.to_string()), b"{\"bytes\":[]}")
            .expect("tamper local fixture");
        assert!(store
            .read_verified::<GraphBytes>(first.id, &first.digest, first.length, 2 * 1024 * 1024,)
            .is_err());
        assert!(store.put_streamed_graph(&graph, 2 * 1024 * 1024).is_err());
    }
}
