use std::future::Future;
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

pub(super) struct FailingFlushProvider {
    pub(super) fail_flush: Arc<AtomicBool>,
    pub(super) cancel_on_ready: Option<runmat_execution::cancellation::CancellationSource>,
}

fn unsupported_provider_op() -> io::Error {
    io::Error::new(io::ErrorKind::Unsupported, "unsupported test provider op")
}

#[async_trait::async_trait(?Send)]
impl runmat_filesystem::FsProvider for FailingFlushProvider {
    fn open(
        &self,
        _path: &Path,
        _flags: &runmat_filesystem::OpenFlags,
    ) -> io::Result<Box<dyn runmat_filesystem::FileHandle>> {
        Ok(Box::new(FailingFlushHandle {
            inner: Cursor::new(Vec::new()),
            fail_flush: self.fail_flush.clone(),
            cancel_on_ready: self.cancel_on_ready.clone(),
        }))
    }

    async fn read(&self, _path: &Path) -> io::Result<Vec<u8>> {
        Err(unsupported_provider_op())
    }

    async fn write(&self, _path: &Path, _data: &[u8]) -> io::Result<()> {
        Err(unsupported_provider_op())
    }

    async fn remove_file(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported_provider_op())
    }

    async fn metadata(&self, _path: &Path) -> io::Result<runmat_filesystem::FsMetadata> {
        Err(unsupported_provider_op())
    }

    async fn symlink_metadata(&self, _path: &Path) -> io::Result<runmat_filesystem::FsMetadata> {
        Err(unsupported_provider_op())
    }

    async fn read_dir(&self, _path: &Path) -> io::Result<Vec<runmat_filesystem::DirEntry>> {
        Err(unsupported_provider_op())
    }

    async fn canonicalize(&self, _path: &Path) -> io::Result<PathBuf> {
        Err(unsupported_provider_op())
    }

    async fn create_dir(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported_provider_op())
    }

    async fn create_dir_all(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported_provider_op())
    }

    async fn remove_dir(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported_provider_op())
    }

    async fn remove_dir_all(&self, _path: &Path) -> io::Result<()> {
        Err(unsupported_provider_op())
    }

    async fn rename(&self, _from: &Path, _to: &Path) -> io::Result<()> {
        Err(unsupported_provider_op())
    }

    async fn set_readonly(&self, _path: &Path, _readonly: bool) -> io::Result<()> {
        Err(unsupported_provider_op())
    }
}

struct FailingFlushHandle {
    inner: Cursor<Vec<u8>>,
    fail_flush: Arc<AtomicBool>,
    cancel_on_ready: Option<runmat_execution::cancellation::CancellationSource>,
}

struct YieldOnce {
    yielded: bool,
}

impl Future for YieldOnce {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

impl Read for FailingFlushHandle {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl Write for FailingFlushHandle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.fail_flush.load(Ordering::SeqCst) {
            Err(io::Error::other("flush failed"))
        } else {
            Ok(())
        }
    }
}

impl Seek for FailingFlushHandle {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

#[async_trait::async_trait(?Send)]
impl runmat_filesystem::FileHandle for FailingFlushHandle {
    async fn flush_async(&mut self) -> io::Result<()> {
        YieldOnce { yielded: false }.await;
        if let Some(source) = &self.cancel_on_ready {
            source.cancel(runmat_execution::CancellationReason::User);
        }
        self.flush()
    }
}
