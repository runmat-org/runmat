//! One bounded producer per application stream, independent of control traffic.
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::Notify;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProcessOutputCompletion {
    NotCaptured,
    Reading,
    EndOfStream,
    ReadFailed(std::io::ErrorKind),
    Transferred,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessOutputSnapshot {
    pub bytes: Box<[u8]>,
    pub observed_bytes: Option<u64>,
    pub truncated: bool,
    pub completion: ProcessOutputCompletion,
}
impl std::fmt::Display for ProcessOutputSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?}, truncated={}: {}",
            self.completion,
            self.truncated,
            String::from_utf8_lossy(&self.bytes)
        )
    }
}

struct State {
    bytes: Vec<u8>,
    observed_bytes: Option<u64>,
    truncated: bool,
    completion: ProcessOutputCompletion,
}

#[derive(Clone)]
pub struct CapturedProcessOutput {
    state: Arc<Mutex<State>>,
    closed: Arc<Notify>,
    limit: usize,
}

/// The unique producer of this bounded diagnostic prefix. A caller which owns
/// the application pipe can also forward bytes without a competing reader.
pub struct ProcessOutputCaptureWriter {
    capture: CapturedProcessOutput,
    finished: bool,
}

impl ProcessOutputCaptureWriter {
    pub fn observe(&mut self, bytes: &[u8]) -> crate::ProcessHostResult<()> {
        if self.finished {
            return Err(crate::ProcessHostError::Protocol(
                "process output capture is already completed".into(),
            ));
        }
        self.capture.push(bytes);
        Ok(())
    }

    pub fn finish(&mut self, outcome: Result<(), std::io::ErrorKind>) {
        if self.finished {
            return;
        }
        self.capture
            .state
            .lock()
            .expect("child output lock poisoned")
            .completion = match outcome {
            Ok(()) => ProcessOutputCompletion::EndOfStream,
            Err(kind) => ProcessOutputCompletion::ReadFailed(kind),
        };
        self.finished = true;
        self.capture.closed.notify_waiters();
    }

    /// Continue consuming the stream after its retained prefix is full, so
    /// application writes cannot block an unrelated control response.
    pub fn drain(mut self, mut reader: impl AsyncRead + Unpin + Send + 'static) {
        tokio::spawn(async move {
            let mut chunk = [0_u8; 8192];
            let completion = loop {
                match reader.read(&mut chunk).await {
                    Ok(0) => break Ok(()),
                    Err(error) => break Err(error.kind()),
                    Ok(read) => {
                        if self.observe(&chunk[..read]).is_err() {
                            break Err(std::io::ErrorKind::InvalidData);
                        }
                    }
                }
            };
            self.finish(completion);
        });
    }
}

impl Drop for ProcessOutputCaptureWriter {
    fn drop(&mut self) {
        self.finish(Err(std::io::ErrorKind::Interrupted));
    }
}
impl std::fmt::Debug for CapturedProcessOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturedProcessOutput")
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}
impl CapturedProcessOutput {
    pub fn new(limit: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                bytes: Vec::new(),
                observed_bytes: Some(0),
                truncated: false,
                completion: ProcessOutputCompletion::NotCaptured,
            })),
            closed: Arc::new(Notify::new()),
            limit,
        }
    }
    pub fn drain(
        &self,
        reader: impl AsyncRead + Unpin + Send + 'static,
    ) -> crate::ProcessHostResult<()> {
        self.begin_capture()?.drain(reader);
        Ok(())
    }

    pub fn begin_capture(&self) -> crate::ProcessHostResult<ProcessOutputCaptureWriter> {
        {
            let mut state = self.state.lock().expect("child output lock poisoned");
            if state.completion != ProcessOutputCompletion::NotCaptured {
                return Err(crate::ProcessHostError::Protocol(
                    "process output capture already has its original reader".into(),
                ));
            }
            state.completion = ProcessOutputCompletion::Reading;
        }
        Ok(ProcessOutputCaptureWriter {
            capture: self.clone(),
            finished: false,
        })
    }
    /// EOF and read failure remain distinct; callers own their timeout policy.
    pub async fn wait_closed(&self) {
        loop {
            let notified = self.closed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self
                .state
                .lock()
                .expect("child output lock poisoned")
                .completion
                != ProcessOutputCompletion::Reading
            {
                return;
            }
            notified.await;
        }
    }
    pub fn completion(&self) -> ProcessOutputCompletion {
        self.state
            .lock()
            .expect("child output lock poisoned")
            .completion
    }
    pub fn snapshot(&self) -> ProcessOutputSnapshot {
        let state = self.state.lock().expect("child output lock poisoned");
        ProcessOutputSnapshot {
            bytes: state.bytes.clone().into_boxed_slice(),
            observed_bytes: state.observed_bytes,
            truncated: state.truncated,
            completion: state.completion,
        }
    }
    /// Move a terminal stream's prefix into its final process-scoped report.
    /// The unique producer has finished; no second payload mirror is retained.
    pub fn take_closed_snapshot(&self) -> crate::ProcessHostResult<ProcessOutputSnapshot> {
        let mut state = self.state.lock().expect("child output lock poisoned");
        if matches!(
            state.completion,
            ProcessOutputCompletion::Reading
                | ProcessOutputCompletion::NotCaptured
                | ProcessOutputCompletion::Transferred
        ) {
            return Err(crate::ProcessHostError::Protocol(
                "process output cannot transfer before its producer finishes".into(),
            ));
        }
        let snapshot = ProcessOutputSnapshot {
            bytes: std::mem::take(&mut state.bytes).into_boxed_slice(),
            observed_bytes: state.observed_bytes,
            truncated: state.truncated,
            completion: state.completion,
        };
        state.completion = ProcessOutputCompletion::Transferred;
        Ok(snapshot)
    }
    pub fn bytes(&self) -> Vec<u8> {
        self.state
            .lock()
            .expect("child output lock poisoned")
            .bytes
            .clone()
    }
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.state.lock().expect("child output lock poisoned").bytes)
            .into_owned()
    }
    fn push(&self, bytes: &[u8]) {
        let mut state = self.state.lock().expect("child output lock poisoned");
        let remaining = self.limit.saturating_sub(state.bytes.len());
        state.truncated |= bytes.len() > remaining;
        state.observed_bytes = state
            .observed_bytes
            .and_then(|n| n.checked_add(bytes.len() as u64));
        state
            .bytes
            .extend_from_slice(&bytes[..bytes.len().min(remaining)]);
    }
}
