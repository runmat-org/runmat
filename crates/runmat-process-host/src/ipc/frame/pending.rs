use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::AsyncWrite;

use super::{frame_header, FrameLimits};
use crate::{ProcessHostError, ProcessHostResult};

/// A framed write whose byte and flush progress survives cancellation of the
/// polling future. The owner must retain this value until Ready before sending
/// another frame on the same writer.
pub struct PendingFrameWrite {
    frame: Vec<u8>,
    written: usize,
    flushed: bool,
    failed: bool,
}

impl PendingFrameWrite {
    pub fn new(payload: Vec<u8>, limits: FrameLimits) -> ProcessHostResult<Self> {
        let header = frame_header(payload.len(), limits)?;
        let mut frame = Vec::with_capacity(4 + payload.len());
        frame.extend_from_slice(&header);
        frame.extend_from_slice(&payload);
        Ok(Self {
            frame,
            written: 0,
            flushed: false,
            failed: false,
        })
    }

    pub fn poll_write(
        &mut self,
        writer: &mut (impl AsyncWrite + Unpin),
        context: &mut Context<'_>,
    ) -> Poll<ProcessHostResult<()>> {
        if self.failed {
            return Poll::Ready(Err(ProcessHostError::Protocol(
                "framed writer failed before settlement".into(),
            )));
        }
        while self.written < self.frame.len() {
            match Pin::new(&mut *writer).poll_write(context, &self.frame[self.written..]) {
                Poll::Ready(Ok(0)) => {
                    self.failed = true;
                    return Poll::Ready(Err(ProcessHostError::Io(std::io::Error::new(
                        std::io::ErrorKind::WriteZero,
                        "framed writer made no progress",
                    ))));
                }
                Poll::Ready(Ok(count)) => self.written += count,
                Poll::Ready(Err(error)) => {
                    self.failed = true;
                    return Poll::Ready(Err(error.into()));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
        if !self.flushed {
            match Pin::new(writer).poll_flush(context) {
                Poll::Ready(Ok(())) => self.flushed = true,
                Poll::Ready(Err(error)) => {
                    self.failed = true;
                    return Poll::Ready(Err(error.into()));
                }
                Poll::Pending => return Poll::Pending,
            }
        }
        Poll::Ready(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::task::{Wake, Waker};

    struct NoopWake;

    impl Wake for NoopWake {
        fn wake(self: Arc<Self>) {}
    }

    #[derive(Default)]
    struct PartialWriter {
        bytes: Vec<u8>,
        first_count: usize,
        write_step: u8,
        flush_pending: bool,
        fail_after_first: bool,
    }

    impl AsyncWrite for PartialWriter {
        fn poll_write(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
            bytes: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            match self.write_step {
                0 => {
                    let count = bytes.len().min(self.first_count);
                    self.bytes.extend_from_slice(&bytes[..count]);
                    self.write_step = 1;
                    Poll::Ready(Ok(count))
                }
                1 => {
                    self.write_step = 2;
                    if self.fail_after_first {
                        Poll::Ready(Err(std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "test transport broke after partial header",
                        )))
                    } else {
                        Poll::Pending
                    }
                }
                _ => {
                    self.bytes.extend_from_slice(bytes);
                    Poll::Ready(Ok(bytes.len()))
                }
            }
        }

        fn poll_flush(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            if !self.flush_pending {
                self.flush_pending = true;
                Poll::Pending
            } else {
                Poll::Ready(Ok(()))
            }
        }

        fn poll_shutdown(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<std::io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    #[test]
    fn partial_header_payload_and_flush_resume_without_duplicate_bytes() {
        let limits = FrameLimits {
            max_message_bytes: 64,
        };
        let payload = b"callback reply".to_vec();
        for first_count in [2, 6] {
            let mut frame = PendingFrameWrite::new(payload.clone(), limits).unwrap();
            let mut writer = PartialWriter {
                first_count,
                ..PartialWriter::default()
            };
            let waker = Waker::from(Arc::new(NoopWake));
            let mut context = Context::from_waker(&waker);
            assert!(matches!(
                frame.poll_write(&mut writer, &mut context),
                Poll::Pending
            ));
            assert_eq!(writer.bytes.len(), first_count);
            assert!(matches!(
                frame.poll_write(&mut writer, &mut context),
                Poll::Pending
            ));
            assert!(matches!(
                frame.poll_write(&mut writer, &mut context),
                Poll::Ready(Ok(()))
            ));
            let mut reader = std::io::Cursor::new(writer.bytes);
            assert_eq!(
                super::super::read_payload_blocking(&mut reader, limits).unwrap(),
                payload
            );
            assert_eq!(reader.position() as usize, reader.get_ref().len());
        }
        assert!(PendingFrameWrite::new(
            Vec::new(),
            FrameLimits {
                max_message_bytes: 0
            }
        )
        .is_err());
        assert!(PendingFrameWrite::new(vec![0; 65], limits).is_err());
    }

    #[test]
    fn io_failure_after_partial_header_is_terminal_without_second_write() {
        let mut frame = PendingFrameWrite::new(
            b"reply".to_vec(),
            FrameLimits {
                max_message_bytes: 64,
            },
        )
        .unwrap();
        let mut writer = PartialWriter {
            first_count: 2,
            fail_after_first: true,
            ..PartialWriter::default()
        };
        let waker = Waker::from(Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        assert!(matches!(
            frame.poll_write(&mut writer, &mut context),
            Poll::Ready(Err(ProcessHostError::Io(_)))
        ));
        let written = writer.bytes.clone();
        let calls = writer.write_step;
        assert!(matches!(
            frame.poll_write(&mut writer, &mut context),
            Poll::Ready(Err(ProcessHostError::Protocol(_)))
        ));
        assert_eq!(writer.bytes, written);
        assert_eq!(writer.write_step, calls);
    }
}
