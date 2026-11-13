//! One bounded reader preserves partially received frames across cancellation.
use runmat_test::protocol::{ProtocolLimits, WorkerRequest};
#[cfg(test)]
use tokio::io::AsyncRead;
use tokio::sync::mpsc;

use crate::{NativeRunnerError, NativeRunnerResult};

pub(super) struct ControlRequests {
    requests: mpsc::Receiver<NativeRunnerResult<WorkerRequest>>,
    frames: Option<mpsc::Receiver<runmat_process_host::ProcessHostResult<Vec<u8>>>>,
    limits: ProtocolLimits,
    reader: Option<tokio::task::JoinHandle<()>>,
}

impl ControlRequests {
    #[cfg(test)]
    pub(super) fn new(
        mut input: impl AsyncRead + Unpin + Send + 'static,
        limits: ProtocolLimits,
    ) -> Self {
        let (sender, requests) = mpsc::channel(1);
        let reader = tokio::spawn(async move {
            loop {
                // Reserve before reading: there is never an extra decoded
                // frame parked outside the one-slot admission bound.
                let Ok(permit) = sender.reserve().await else {
                    break;
                };
                let request = crate::transport::read_request(&mut input, limits).await;
                let terminal = request.is_err();
                permit.send(request);
                if terminal {
                    break;
                }
            }
        });
        Self {
            requests,
            frames: None,
            limits,
            reader: Some(reader),
        }
    }

    pub(super) fn from_frames(
        frames: mpsc::Receiver<runmat_process_host::ProcessHostResult<Vec<u8>>>,
        limits: ProtocolLimits,
    ) -> Self {
        let (_, requests) = mpsc::channel(1);
        Self {
            requests,
            frames: Some(frames),
            limits,
            reader: None,
        }
    }

    pub(super) async fn next(&mut self) -> NativeRunnerResult<WorkerRequest> {
        if let Some(frames) = &mut self.frames {
            let payload = frames
                .recv()
                .await
                .ok_or_else(|| NativeRunnerError::Protocol("native parent reader stopped".into()))?
                .map_err(|error| NativeRunnerError::Protocol(error.to_string()))?;
            return runmat_test::protocol::decode_request(&payload, self.limits)
                .map_err(|error| NativeRunnerError::Protocol(error.to_string()));
        }
        self.requests.recv().await.ok_or_else(|| {
            NativeRunnerError::Protocol("native worker control reader stopped".into())
        })?
    }
}

impl Drop for ControlRequests {
    fn drop(&mut self) {
        if let Some(reader) = &self.reader {
            reader.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::Pin;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use std::task::{Context, Poll};
    use tokio::io::{AsyncWriteExt, ReadBuf};

    struct CountedInput {
        inner: tokio::io::DuplexStream,
        read: Arc<AtomicUsize>,
    }
    impl AsyncRead for CountedInput {
        fn poll_read(
            self: Pin<&mut Self>,
            context: &mut Context<'_>,
            output: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            let this = self.get_mut();
            let before = output.filled().len();
            let result = Pin::new(&mut this.inner).poll_read(context, output);
            this.read
                .fetch_add(output.filled().len() - before, Ordering::Release);
            result
        }
    }

    #[tokio::test]
    async fn cancelled_consumer_preserves_the_original_partial_control_frame() {
        let (mut writer, reader) = tokio::io::duplex(4096);
        let read = Arc::new(AtomicUsize::new(0));
        let limits = ProtocolLimits::default();
        let mut requests = ControlRequests::new(
            CountedInput {
                inner: reader,
                read: read.clone(),
            },
            limits,
        );
        let frame =
            runmat_test_runner::worker::encode_frame(&WorkerRequest::Shutdown, limits).unwrap();
        writer.write_all(&frame[..2]).await.unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while read.load(Ordering::Acquire) != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("reader must consume the original partial header");
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(1), requests.next())
                .await
                .is_err()
        );
        writer.write_all(&frame[2..]).await.unwrap();
        assert!(matches!(
            requests.next().await.unwrap(),
            WorkerRequest::Shutdown
        ));
        writer.shutdown().await.unwrap();
        assert!(requests.next().await.is_err());
    }
}
