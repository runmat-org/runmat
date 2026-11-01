//! Private process control, with explicit pipe fixtures for protocol tests.
use std::pin::Pin;
use std::task::{Context, Poll};

use runmat_process_host::ipc::private_control::{ControlReader, ControlWriter};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

pub(super) enum FramedReader {
    Private(ControlReader),
    #[cfg(test)]
    Fixture(tokio::process::ChildStdout),
}

pub(super) enum FramedWriter {
    Private(ControlWriter),
    #[cfg(test)]
    Fixture(tokio::process::ChildStdin),
}

impl AsyncRead for FramedReader {
    fn poll_read(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Private(reader) => Pin::new(reader).poll_read(context, buffer),
            #[cfg(test)]
            Self::Fixture(reader) => Pin::new(reader).poll_read(context, buffer),
        }
    }
}

impl AsyncWrite for FramedWriter {
    fn poll_write(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            Self::Private(writer) => Pin::new(writer).poll_write(context, bytes),
            #[cfg(test)]
            Self::Fixture(writer) => Pin::new(writer).poll_write(context, bytes),
        }
    }
    fn poll_flush(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Private(writer) => Pin::new(writer).poll_flush(context),
            #[cfg(test)]
            Self::Fixture(writer) => Pin::new(writer).poll_flush(context),
        }
    }
    fn poll_shutdown(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Private(writer) => Pin::new(writer).poll_shutdown(context),
            #[cfg(test)]
            Self::Fixture(writer) => Pin::new(writer).poll_shutdown(context),
        }
    }
}
