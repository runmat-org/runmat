//! Physical native-host ownership, independent of callable/value routing.
//!
//! A trusted host owns its libraries and pointers on one OS thread. Only the
//! existing bounded wire protocol crosses that boundary; evaluator Values,
//! GC roots, and callback futures remain on their original executor.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use runmat_process_host::{BootstrapBoundChild, ContainedChildRetirement, ProcessExit};
use tokio::io::{
    AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, DuplexStream, ReadBuf, ReadHalf, WriteHalf,
};

use super::client::NativeChildResources;
mod thread_owner;
use thread_owner::ThreadOwner;

// Match the CLI command stack allowance instead of moving existing trusted
// native calls onto Rust's much smaller default worker stack. Stack address
// space is reserved; untouched pages are not duplicated evaluator memory.
const NATIVE_THREAD_STACK_BYTES: usize = 64 * 1024 * 1024;

pub(super) enum HostOwner {
    Process(Box<BootstrapBoundChild<NativeChildResources>>),
    Thread(ThreadOwner),
}

pub(super) struct HostExit {
    pub(super) success: bool,
    pub(super) code: Option<i32>,
}

impl From<ProcessExit> for HostExit {
    fn from(exit: ProcessExit) -> Self {
        Self {
            success: exit.success,
            code: exit.code,
        }
    }
}

impl HostOwner {
    pub(super) fn lease(&self) -> &NativeChildResources {
        match self {
            Self::Process(child) => child.lease(),
            Self::Thread(owner) => &owner.resources,
        }
    }

    pub(super) fn is_thread(&self) -> bool {
        matches!(self, Self::Thread(_))
    }

    pub(super) fn stderr(&self) -> String {
        match self {
            Self::Process(child) => child.captured_stderr().text(),
            Self::Thread(owner) => owner.diagnostic(),
        }
    }

    pub(super) fn try_wait(&mut self) -> Result<Option<HostExit>, String> {
        match self {
            Self::Process(child) => child
                .try_wait()
                .map(|exit| exit.map(HostExit::from))
                .map_err(|error| error.to_string()),
            Self::Thread(owner) => owner.try_wait(),
        }
    }

    pub(super) async fn wait(&mut self) -> Result<HostExit, String> {
        match self {
            Self::Process(child) => child
                .wait()
                .await
                .map(HostExit::from)
                .map_err(|error| error.to_string()),
            Self::Thread(owner) => owner.wait().await,
        }
    }

    /// A thread cannot be killed safely. Only joining its actual exit proves
    /// retirement; an active native call retains its storage and reports an
    /// unverified owner instead of fabricating process-containment evidence.
    pub(super) async fn retire(&mut self) -> Result<Option<ContainedChildRetirement>, String> {
        match self {
            Self::Process(child) => child
                .terminate_contained_child()
                .await
                .map(Some)
                .map_err(|error| error.to_string()),
            Self::Thread(_) => match self.try_wait()? {
                Some(_) => Ok(None),
                None => Err(
                    "trusted native host thread has not exited; its resources remain owned".into(),
                ),
            },
        }
    }
}

pub(super) enum HostReader {
    Process(runmat_process_host::ipc::private_control::ControlReader),
    Thread(ReadHalf<DuplexStream>),
    Closed,
}

pub(super) enum HostWriter {
    Process(runmat_process_host::ipc::private_control::ControlWriter),
    Thread(WriteHalf<DuplexStream>),
    Closed,
}

impl AsyncRead for HostReader {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Process(reader) => Pin::new(reader).poll_read(context, buffer),
            Self::Thread(reader) => Pin::new(reader).poll_read(context, buffer),
            Self::Closed => Poll::Ready(Ok(())),
        }
    }
}

impl AsyncWrite for HostWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            Self::Process(writer) => Pin::new(writer).poll_write(context, bytes),
            Self::Thread(writer) => Pin::new(writer).poll_write(context, bytes),
            Self::Closed => Poll::Ready(Err(std::io::ErrorKind::BrokenPipe.into())),
        }
    }
    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Process(writer) => Pin::new(writer).poll_flush(context),
            Self::Thread(writer) => Pin::new(writer).poll_flush(context),
            Self::Closed => Poll::Ready(Ok(())),
        }
    }
    fn poll_shutdown(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Process(writer) => Pin::new(writer).poll_shutdown(context),
            Self::Thread(writer) => Pin::new(writer).poll_shutdown(context),
            Self::Closed => Poll::Ready(Ok(())),
        }
    }
}

pub(super) fn spawn_thread(
    resources: NativeChildResources,
    frontend: std::path::PathBuf,
) -> Result<(HostOwner, HostReader, HostWriter), String> {
    // Bounded streaming transport: a large legal frame never requires a
    // second unbounded message queue while the evaluator is suspended.
    let (client, server) = tokio::io::duplex(64 * 1024);
    let (reader, writer) = tokio::io::split(client);
    let resources = Arc::new(resources);
    let thread_resources = Arc::clone(&resources);
    let thread = std::thread::Builder::new()
        .name("runmat-native-ffi".into())
        .stack_size(NATIVE_THREAD_STACK_BYTES)
        .spawn(move || {
            let (reader, writer) = tokio::io::split(server);
            super::server::run_thread_host(
                BlockingReader(reader),
                BlockingWriter(writer),
                &thread_resources,
                frontend,
            )
        })
        .map_err(|error| format!("could not start native host thread: {error}"))?;
    Ok((
        HostOwner::Thread(ThreadOwner::new(thread, resources)?),
        HostReader::Thread(reader),
        HostWriter::Thread(writer),
    ))
}

// These adapters block only the native host and its input reader. They never
// drive a RunMat callback or enter the evaluator's asynchronous executor.
struct BlockingReader(ReadHalf<DuplexStream>);
impl std::io::Read for BlockingReader {
    fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
        pollster::block_on(self.0.read(bytes))
    }
}

struct BlockingWriter(WriteHalf<DuplexStream>);
impl std::io::Write for BlockingWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        pollster::block_on(self.0.write(bytes))
    }
    fn flush(&mut self) -> std::io::Result<()> {
        pollster::block_on(self.0.flush())
    }
}
