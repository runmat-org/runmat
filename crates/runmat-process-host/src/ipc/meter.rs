//! Count actual pipe I/O, including partial frames and failed writes. Overflow
//! makes evidence unavailable instead of silently wrapping or clamping it.
use std::io;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

#[derive(Clone, Default)]
pub struct IpcByteCounter {
    state: Arc<CounterState>,
}

#[derive(Default)]
struct CounterState {
    bytes: AtomicU64,
    overflowed: AtomicBool,
}

impl IpcByteCounter {
    pub fn bytes(&self) -> Option<u64> {
        if self.state.overflowed.load(Ordering::Acquire) {
            None
        } else {
            Some(self.state.bytes.load(Ordering::Acquire))
        }
    }

    fn add(&self, amount: usize) -> io::Result<()> {
        let observed = u64::try_from(amount).ok().and_then(|amount| {
            self.state
                .bytes
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                    value.checked_add(amount)
                })
                .ok()
        });
        if observed.is_none() {
            self.state.overflowed.store(true, Ordering::Release);
            return Err(io::Error::other("IPC byte accounting overflowed"));
        }
        Ok(())
    }
}

pub struct MeteredIo<T> {
    inner: T,
    counter: IpcByteCounter,
}

impl<T> MeteredIo<T> {
    pub fn new(inner: T, counter: IpcByteCounter) -> Self {
        Self { inner, counter }
    }
}

impl<T: AsyncRead + Unpin> AsyncRead for MeteredIo<T> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let before = buf.filled().len();
        let result = Pin::new(&mut this.inner).poll_read(cx, buf);
        if let Err(error) = this.counter.add(buf.filled().len() - before) {
            return Poll::Ready(Err(error));
        }
        result
    }
}

impl<T: AsyncWrite + Unpin> AsyncWrite for MeteredIo<T> {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        match Pin::new(&mut this.inner).poll_write(cx, buf) {
            Poll::Ready(Ok(written)) => match this.counter.add(written) {
                Ok(()) => Poll::Ready(Ok(written)),
                Err(error) => Poll::Ready(Err(error)),
            },
            result => result,
        }
    }
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_flush(cx)
    }
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().inner).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests;
