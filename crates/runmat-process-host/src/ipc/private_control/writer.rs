//! One owner serializes blocking console and asynchronous control frames.
//! Accepted frames survive cancellation of their caller's reply future.
use std::fmt;
use std::io::Write;
use std::sync::{mpsc, Arc, Mutex};

use tokio::sync::{oneshot, Notify};

use crate::ipc::{write_payload_blocking, FrameLimits};
use crate::{ProcessHostError, ProcessHostResult};

use super::BlockingControlWriter;

#[derive(Clone, Debug)]
pub struct ControlWriteError(Arc<ProcessHostError>);

impl fmt::Display for ControlWriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.0.as_ref(), formatter)
    }
}
impl std::error::Error for ControlWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

type WriteResult = Result<(), ControlWriteError>;

enum Reply {
    Blocking(mpsc::SyncSender<WriteResult>),
    Async(oneshot::Sender<WriteResult>),
}
impl Reply {
    fn finish(self, result: WriteResult) {
        match self {
            Self::Blocking(reply) => {
                let _ = reply.send(result);
            }
            Self::Async(reply) => {
                let _ = reply.send(result);
            }
        }
    }
}

struct Frame {
    bytes: Vec<u8>,
    reply: Reply,
}

struct State {
    sender: mpsc::SyncSender<Frame>,
    space: Arc<Notify>,
    failure: Arc<Mutex<Option<ControlWriteError>>>,
    limits: FrameLimits,
}

/// There is one queued frame and one in-flight write, never an unbounded
/// producer queue. The writer owns its descriptor; producers move encoded
/// buffers into it instead of cloning them or sharing descriptor locks.
#[derive(Clone)]
pub struct ControlFrameWriter(Arc<State>);

impl ControlFrameWriter {
    pub fn new(writer: BlockingControlWriter, limits: FrameLimits) -> ProcessHostResult<Self> {
        Self::start(writer, limits)
    }

    fn start(
        mut writer: impl Write + Send + 'static,
        limits: FrameLimits,
    ) -> ProcessHostResult<Self> {
        if limits.max_message_bytes == 0 {
            return Err(ProcessHostError::Protocol(
                "control frame bound must be positive".into(),
            ));
        }
        let (sender, receiver) = mpsc::sync_channel::<Frame>(1);
        let space = Arc::new(Notify::new());
        let failure = Arc::new(Mutex::new(None));
        let worker_space = space.clone();
        let worker_failure = failure.clone();
        std::thread::Builder::new()
            .name("runmat-control-writer".into())
            .spawn(move || {
                while let Ok(frame) = receiver.recv() {
                    worker_space.notify_waiters();
                    let result = write_payload_blocking(&mut writer, &frame.bytes, limits)
                        .map_err(|error| ControlWriteError(Arc::new(error)));
                    if let Err(error) = &result {
                        *worker_failure
                            .lock()
                            .expect("control writer failure lock poisoned") = Some(error.clone());
                    }
                    let failed = result.is_err();
                    frame.reply.finish(result);
                    if failed {
                        break;
                    }
                }
                // Drop queued reply owners before waking producers waiting for space.
                drop(receiver);
                worker_space.notify_waiters();
            })?;
        Ok(Self(Arc::new(State {
            sender,
            space,
            failure,
            limits,
        })))
    }

    fn validate(&self, bytes: &[u8]) -> WriteResult {
        if bytes.len() > self.0.limits.max_message_bytes as usize {
            return Err(ControlWriteError(Arc::new(ProcessHostError::Protocol(
                "control frame exceeds its declared bound".into(),
            ))));
        }
        if let Some(error) = self
            .0
            .failure
            .lock()
            .expect("control writer failure lock poisoned")
            .as_ref()
        {
            return Err(error.clone());
        }
        Ok(())
    }

    fn stopped(&self) -> ControlWriteError {
        self.0
            .failure
            .lock()
            .expect("control writer failure lock poisoned")
            .clone()
            .unwrap_or_else(|| {
                ControlWriteError(Arc::new(ProcessHostError::Io(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "control writer stopped",
                ))))
            })
    }

    /// Synchronous runtime console sinks apply ordinary bounded backpressure.
    /// No Tokio blocking API is called from the runtime's executor thread.
    pub fn write_blocking(&self, bytes: Vec<u8>) -> WriteResult {
        self.validate(&bytes)?;
        let (reply, result) = mpsc::sync_channel(1);
        self.0
            .sender
            .send(Frame {
                bytes,
                reply: Reply::Blocking(reply),
            })
            .map_err(|_| self.stopped())?;
        result.recv().map_err(|_| self.stopped())?
    }

    /// Cancelling before enqueue writes nothing. Cancelling after enqueue
    /// releases only the reply observer: the original full frame stays owned.
    pub async fn write_owned(&self, bytes: Vec<u8>) -> WriteResult {
        self.validate(&bytes)?;
        let (reply, result) = oneshot::channel();
        let mut frame = Frame {
            bytes,
            reply: Reply::Async(reply),
        };
        loop {
            let space = self.0.space.notified();
            tokio::pin!(space);
            space.as_mut().enable();
            match self.0.sender.try_send(frame) {
                Ok(()) => break,
                Err(mpsc::TrySendError::Full(retained)) => frame = retained,
                Err(mpsc::TrySendError::Disconnected(_)) => return Err(self.stopped()),
            }
            space.await;
        }
        result.await.map_err(|_| self.stopped())?
    }
}

#[cfg(test)]
mod tests;
