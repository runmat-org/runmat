//! Local control transport for children which can write application stdout.
//!
//! Framing and session authentication belong to the caller. These two one-way
//! channels keep application bytes out of that protocol and make closing the
//! driver's writer observable as EOF by the blocking host reader on both OSes.

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

mod writer;
pub use writer::{ControlFrameWriter, ControlWriteError};

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;
#[cfg(unix)]
use unix as platform;
#[cfg(windows)]
use windows as platform;

pub use platform::{
    BlockingReader as BlockingControlReader, BlockingWriter as BlockingControlWriter,
};

pub(crate) const BOOTSTRAP_ENV: &str = "RUNMAT_PRIVATE_CONTROL_ENDPOINT";

/// Consume the child-only bootstrap address before loading user libraries.
/// Missing configuration is an error; stdout is never a control fallback.
pub fn connect_host_blocking() -> io::Result<BlockingControl> {
    host_endpoint()?.connect_blocking()
}

/// Async hosts use their own executor for control reads and cancellation.
pub async fn connect_host() -> io::Result<ControlConnection> {
    host_endpoint()?.connect().await
}

fn host_endpoint() -> io::Result<PrivateControlEndpoint> {
    let encoded = std::env::var(BOOTSTRAP_ENV)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    std::env::remove_var(BOOTSTRAP_ENV);
    PrivateControlEndpoint::decode(&encoded)
}

/// A bootstrap address contains no authentication authority. Callers must run
/// the existing authenticated session handshake before accepting any messages.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct PrivateControlEndpoint {
    request: String,
    response: String,
}

impl PrivateControlEndpoint {
    pub fn encode(&self) -> io::Result<String> {
        serde_json::to_string(self).map_err(io::Error::other)
    }

    pub fn decode(encoded: &str) -> io::Result<Self> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireEndpoint {
            request: String,
            response: String,
        }
        let wire: WireEndpoint = serde_json::from_str(encoded)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        let endpoint = Self {
            request: wire.request,
            response: wire.response,
        };
        platform::validate_endpoint(&endpoint)?;
        Ok(endpoint)
    }

    /// Connect on the host's blocking protocol thread, not an async executor.
    pub fn connect_blocking(&self) -> io::Result<BlockingControl> {
        platform::validate_endpoint(self)?;
        platform::connect_blocking(self)
    }

    pub async fn connect(&self) -> io::Result<ControlConnection> {
        platform::validate_endpoint(self)?;
        platform::connect(self).await
    }
}

pub struct BlockingControl {
    pub reader: platform::BlockingReader,
    pub writer: platform::BlockingWriter,
}

/// Owns the private listening namespace until both channels have connected.
/// Dropping before connection closes the listeners and removes the namespace.
pub struct PrivateControlListener(platform::Listener);

impl PrivateControlListener {
    pub fn bind() -> io::Result<Self> {
        platform::Listener::bind().map(Self)
    }

    pub fn endpoint(&self) -> &PrivateControlEndpoint {
        self.0.endpoint()
    }

    /// A caller must bound this bootstrap with its child-start deadline and
    /// retain the process owner if the future is cancelled or fails.
    pub async fn accept(self) -> io::Result<ControlConnection> {
        self.0.accept().await
    }
}

pub struct ControlConnection {
    pub reader: ControlReader,
    pub writer: ControlWriter,
}

pub struct ControlReader(platform::Reader);

/// Shutdown drops the exact writer after flushing it. In particular, Windows
/// named-pipe EOF must not wait for the entire client object to be dropped.
pub struct ControlWriter(Option<platform::Writer>);

impl AsyncRead for ControlReader {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().0).poll_read(context, buffer)
    }
}

impl AsyncWrite for ControlWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut().0.as_mut() {
            Some(writer) => Pin::new(writer).poll_write(context, bytes),
            None => Poll::Ready(Err(io::ErrorKind::BrokenPipe.into())),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut().0.as_mut() {
            Some(writer) => Pin::new(writer).poll_flush(context),
            None => Poll::Ready(Ok(())),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        if let Some(writer) = this.0.as_mut() {
            match Pin::new(writer).poll_shutdown(context) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                Poll::Ready(Ok(())) => {}
            }
        }
        this.0.take();
        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod tests;
