use async_trait::async_trait;
use futures::channel::oneshot;
use runmat_filesystem::{
    FsPublicationDisposition, FsPublicationFailure, FsResourceContract, FsResourceLease,
    FsResourceRevision, FsStagedOutput,
};
use std::io;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

mod provider;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum PausePoint {
    Write,
    Flush,
    Published,
}

pub(super) struct StageFixture {
    pub(super) provider: Arc<PausedProvider>,
    pub(super) started: oneshot::Receiver<()>,
    pub(super) resume: oneshot::Sender<()>,
    pub(super) published: Arc<Mutex<Option<Vec<u8>>>>,
    pub(super) drops: Arc<AtomicUsize>,
    pub(super) lease: Arc<RevocableLease>,
}

impl StageFixture {
    pub(super) fn new(point: PausePoint) -> Self {
        let (resume, gate) = oneshot::channel();
        let (start, started) = oneshot::channel();
        let published = Arc::new(Mutex::new(None));
        let drops = Arc::new(AtomicUsize::new(0));
        let lease = RevocableLease::live();
        Self {
            provider: Arc::new(PausedProvider {
                point,
                gate: Mutex::new(Some(gate)),
                started: Mutex::new(Some(start)),
                published: published.clone(),
                drops: drops.clone(),
                lease: lease.clone(),
            }),
            started,
            resume,
            published,
            drops,
            lease,
        }
    }
}

pub(super) struct RevocableLease(AtomicBool);

impl RevocableLease {
    fn live() -> Arc<Self> {
        Arc::new(Self(AtomicBool::new(true)))
    }
    pub(super) fn revoke(&self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

impl FsResourceLease for RevocableLease {
    fn contract(&self) -> FsResourceContract {
        FsResourceContract {
            revision: FsResourceRevision::for_provider("figure.persistence.stage", b"v1"),
            opened_input_handle: None,
            opened_output_handle: None,
        }
    }
    fn is_valid(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

pub(super) struct PausedProvider {
    point: PausePoint,
    gate: Mutex<Option<oneshot::Receiver<()>>>,
    started: Mutex<Option<oneshot::Sender<()>>>,
    published: Arc<Mutex<Option<Vec<u8>>>>,
    drops: Arc<AtomicUsize>,
    lease: Arc<RevocableLease>,
}

struct PausedStage {
    point: PausePoint,
    gate: Option<oneshot::Receiver<()>>,
    started: Option<oneshot::Sender<()>>,
    bytes: Vec<u8>,
    published: Arc<Mutex<Option<Vec<u8>>>>,
    drops: Arc<AtomicUsize>,
}

impl PausedStage {
    async fn pause(&mut self, point: PausePoint) -> io::Result<()> {
        if self.point != point {
            return Ok(());
        }
        if let Some(started) = self.started.take() {
            let _ = started.send(());
        }
        self.gate
            .take()
            .expect("selected stage pauses once")
            .await
            .map_err(|_| io::Error::other("stage pause abandoned"))
    }
}

impl Drop for PausedStage {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

#[async_trait(?Send)]
impl FsStagedOutput for PausedStage {
    async fn write_chunk(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.pause(PausePoint::Write).await?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    async fn flush_and_sync(&mut self) -> io::Result<()> {
        self.pause(PausePoint::Flush).await
    }
    async fn publish(mut self: Box<Self>) -> Result<(), FsPublicationFailure> {
        *self.published.lock().unwrap() = Some(std::mem::take(&mut self.bytes));
        self.pause(PausePoint::Published).await.map_err(|source| {
            FsPublicationFailure::new(FsPublicationDisposition::Indeterminate, source)
        })
    }
}
