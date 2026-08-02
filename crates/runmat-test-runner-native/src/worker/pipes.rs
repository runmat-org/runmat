use std::sync::{Arc, Mutex};

use tokio::io::{AsyncRead, AsyncReadExt};

#[derive(Clone, Debug)]
pub(super) struct CapturedStderr {
    bytes: Arc<Mutex<Vec<u8>>>,
    limit: usize,
}

impl CapturedStderr {
    pub fn new(limit: usize) -> Self {
        Self {
            bytes: Arc::new(Mutex::new(Vec::new())),
            limit,
        }
    }

    pub fn drain(&self, mut reader: impl AsyncRead + Unpin + Send + 'static) {
        let capture = self.clone();
        tokio::spawn(async move {
            let mut chunk = [0_u8; 8192];
            loop {
                match reader.read(&mut chunk).await {
                    Ok(0) | Err(_) => break,
                    Ok(read) => capture.push(&chunk[..read]),
                }
            }
        });
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.bytes.lock().expect("worker stderr lock poisoned"))
            .into_owned()
    }

    fn push(&self, bytes: &[u8]) {
        let mut capture = self.bytes.lock().expect("worker stderr lock poisoned");
        let remaining = self.limit.saturating_sub(capture.len());
        capture.extend_from_slice(&bytes[..bytes.len().min(remaining)]);
    }
}
