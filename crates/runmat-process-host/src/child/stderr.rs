use std::sync::{Arc, Mutex};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::Notify;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StderrCompletion {
    NotCaptured,
    Reading,
    EndOfStream,
    ReadFailed(std::io::ErrorKind),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StderrSnapshot {
    pub bytes: Box<[u8]>,
    pub observed_bytes: Option<u64>,
    pub truncated: bool,
    pub completion: StderrCompletion,
}
impl std::fmt::Display for StderrSnapshot {
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
    completion: StderrCompletion,
}

#[derive(Clone)]
pub struct CapturedStderr {
    state: Arc<Mutex<State>>,
    closed: Arc<Notify>,
    limit: usize,
}
impl std::fmt::Debug for CapturedStderr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CapturedStderr")
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}
impl CapturedStderr {
    pub fn new(limit: usize) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                bytes: Vec::new(),
                observed_bytes: Some(0),
                truncated: false,
                completion: StderrCompletion::NotCaptured,
            })),
            closed: Arc::new(Notify::new()),
            limit,
        }
    }
    pub fn drain(
        &self,
        mut reader: impl AsyncRead + Unpin + Send + 'static,
    ) -> crate::ProcessHostResult<()> {
        {
            let mut state = self.state.lock().expect("child stderr lock poisoned");
            if state.completion != StderrCompletion::NotCaptured {
                return Err(crate::ProcessHostError::Protocol(
                    "stderr capture already has its original reader".into(),
                ));
            }
            state.completion = StderrCompletion::Reading;
        }
        let capture = self.clone();
        tokio::spawn(async move {
            let mut chunk = [0_u8; 8192];
            let completion = loop {
                match reader.read(&mut chunk).await {
                    Ok(0) => break StderrCompletion::EndOfStream,
                    Err(error) => break StderrCompletion::ReadFailed(error.kind()),
                    Ok(read) => capture.push(&chunk[..read]),
                }
            };
            capture
                .state
                .lock()
                .expect("child stderr lock poisoned")
                .completion = completion;
            capture.closed.notify_waiters();
        });
        Ok(())
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
                .expect("child stderr lock poisoned")
                .completion
                != StderrCompletion::Reading
            {
                return;
            }
            notified.await;
        }
    }
    pub fn completion(&self) -> StderrCompletion {
        self.state
            .lock()
            .expect("child stderr lock poisoned")
            .completion
    }
    pub fn snapshot(&self) -> StderrSnapshot {
        let state = self.state.lock().expect("child stderr lock poisoned");
        StderrSnapshot {
            bytes: state.bytes.clone().into_boxed_slice(),
            observed_bytes: state.observed_bytes,
            truncated: state.truncated,
            completion: state.completion,
        }
    }
    pub fn bytes(&self) -> Vec<u8> {
        self.state
            .lock()
            .expect("child stderr lock poisoned")
            .bytes
            .clone()
    }
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.state.lock().expect("child stderr lock poisoned").bytes)
            .into_owned()
    }
    fn push(&self, bytes: &[u8]) {
        let mut state = self.state.lock().expect("child stderr lock poisoned");
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
