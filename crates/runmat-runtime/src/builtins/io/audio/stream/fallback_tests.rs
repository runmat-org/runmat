use super::*;
use async_trait::async_trait;
use futures::executor::block_on;
use runmat_filesystem::{DirEntry, FileHandle, FsMetadata, FsProvider, OpenFlags};
use std::io::{self, Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const PCM16: &[u8] = &[
    0x52, 0x49, 0x46, 0x46, 0x28, 0, 0, 0, 0x57, 0x41, 0x56, 0x45, 0x66, 0x6d, 0x74, 0x20, 0x10, 0,
    0, 0, 1, 0, 1, 0, 0x40, 0x1f, 0, 0, 0x80, 0x3e, 0, 0, 2, 0, 16, 0, 0x64, 0x61, 0x74, 0x61, 4,
    0, 0, 0, 0, 0, 0, 0x40,
];

#[derive(Default)]
struct Evidence {
    opens: AtomicUsize,
    drops: AtomicUsize,
    reads: AtomicUsize,
}

struct UnsupportedSeekHandle {
    bytes: Cursor<Vec<u8>>,
    evidence: Arc<Evidence>,
    seek_kind: io::ErrorKind,
}

impl Read for UnsupportedSeekHandle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.bytes.read(buf)
    }
}
impl Write for UnsupportedSeekHandle {
    fn write(&mut self, _: &[u8]) -> io::Result<usize> {
        Err(unsupported())
    }
    fn flush(&mut self) -> io::Result<()> {
        Err(unsupported())
    }
}
impl Seek for UnsupportedSeekHandle {
    fn seek(&mut self, _: SeekFrom) -> io::Result<u64> {
        Err(io::Error::new(self.seek_kind, "seek failed"))
    }
}
impl FileHandle for UnsupportedSeekHandle {}
impl Drop for UnsupportedSeekHandle {
    fn drop(&mut self) {
        self.evidence.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct ReadOnlyProvider {
    evidence: Arc<Evidence>,
    fail_read: bool,
    seek_kind: io::ErrorKind,
}

fn unsupported() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "seek unsupported")
}
fn unexpected() -> io::Error {
    io::Error::other("unexpected filesystem effect")
}

#[async_trait(?Send)]
impl FsProvider for ReadOnlyProvider {
    fn path_namespace(&self) -> runmat_filesystem::FsPathNamespace {
        runmat_filesystem::FsPathNamespace::VirtualRooted
    }

    fn open(&self, _: &Path, _: &OpenFlags) -> io::Result<Box<dyn FileHandle>> {
        self.evidence.opens.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(UnsupportedSeekHandle {
            bytes: Cursor::new(PCM16.to_vec()),
            evidence: self.evidence.clone(),
            seek_kind: self.seek_kind,
        }))
    }
    async fn read(&self, _: &Path) -> io::Result<Vec<u8>> {
        self.evidence.reads.fetch_add(1, Ordering::SeqCst);
        assert_eq!(
            self.evidence.drops.load(Ordering::SeqCst),
            1,
            "seek handle must retire before provider read"
        );
        if self.fail_read {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "source denied",
            ))
        } else {
            Ok(PCM16.to_vec())
        }
    }
    async fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(unexpected())
    }
    async fn remove_file(&self, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unexpected())
    }
    async fn symlink_metadata(&self, _: &Path) -> io::Result<FsMetadata> {
        Err(unexpected())
    }
    async fn read_dir(&self, _: &Path) -> io::Result<Vec<DirEntry>> {
        Err(unexpected())
    }
    async fn canonicalize(&self, _: &Path) -> io::Result<PathBuf> {
        Err(unexpected())
    }
    async fn create_dir(&self, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn create_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn remove_dir(&self, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn remove_dir_all(&self, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(unexpected())
    }
    async fn set_readonly(&self, _: &Path, _: bool) -> io::Result<()> {
        Err(unexpected())
    }
}

fn run(
    seek_kind: io::ErrorKind,
    fail_read: bool,
) -> (Result<DecodedAudio, RuntimeError>, Arc<Evidence>) {
    let evidence = Arc::new(Evidence::default());
    assert_eq!(
        PCM16.len(),
        48,
        "two complete PCM16 frames after the header"
    );
    let input = FsInputLease {
        provider: Arc::new(ReadOnlyProvider {
            evidence: evidence.clone(),
            fail_read,
            seek_kind,
        }),
        path: PathBuf::from("/audio.wav"),
        resource: None,
    };
    let mut cooperation = AudioCooperation::new(None, AUDIOREAD_BUILTIN_NAME);
    let result = block_on(decode_wave_stream(
        &input,
        AudioreadOptions {
            range: Some(AudioSampleRange {
                first: 2,
                last: None,
            }),
            datatype: AudioReadDataType::Native,
        },
        &mut cooperation,
    ));
    (result, evidence)
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn unsupported_seek_falls_back_once_after_handle_retirement() {
    let (result, evidence) = run(io::ErrorKind::Unsupported, false);
    let decoded = result.expect("read-only provider fallback");
    assert_eq!(decoded.samples.shape(), &[1, 1]);
    assert_eq!(decoded.samples.materialize_f64(), vec![16384.0]);
    assert_eq!(evidence.opens.load(Ordering::SeqCst), 1);
    assert_eq!(evidence.reads.load(Ordering::SeqCst), 1);
    assert_eq!(evidence.drops.load(Ordering::SeqCst), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn fallback_read_operational_failure_keeps_io_cause_without_replay() {
    let (result, evidence) = run(io::ErrorKind::Unsupported, true);
    let error = result.expect_err("read denied");
    assert_eq!(error.identifier(), Some("RunMat:audioread:Io"));
    assert!(error.message().contains("source denied"));
    assert_eq!(evidence.opens.load(Ordering::SeqCst), 1);
    assert_eq!(evidence.reads.load(Ordering::SeqCst), 1);
    assert_eq!(evidence.drops.load(Ordering::SeqCst), 1);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn operational_seek_failure_does_not_trigger_whole_provider_read() {
    let (result, evidence) = run(io::ErrorKind::PermissionDenied, false);
    let error = result.expect_err("seek denied");
    assert_eq!(error.identifier(), Some("RunMat:audioread:Io"));
    assert!(error.message().contains("seek failed"));
    assert_eq!(evidence.opens.load(Ordering::SeqCst), 1);
    assert_eq!(evidence.reads.load(Ordering::SeqCst), 0);
    assert_eq!(evidence.drops.load(Ordering::SeqCst), 1);
}
