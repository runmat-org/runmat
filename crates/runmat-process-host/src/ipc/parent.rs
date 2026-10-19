//! Lifetime of a one-request isolated worker, independent of its async executor.
use crate::ProcessHostResult;
use std::io::Read;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

pub const WORKER_PARENT_MONITOR_STACK_BYTES: u64 = 64 * 1024;

/// Only for a one-request worker process. EOF, input error or unexpected trailing
/// data ends the worker root. A source loop need not yield for parent loss to act.
pub struct WorkerParentGuard {
    armed: Arc<AtomicBool>,
}

impl WorkerParentGuard {
    pub fn start(reader: impl Read + Send + 'static) -> ProcessHostResult<Self> {
        let armed = Arc::new(AtomicBool::new(true));
        let authority = Arc::clone(&armed);
        std::thread::Builder::new()
            .name("runmat-worker-parent".into())
            .stack_size(WORKER_PARENT_MONITOR_STACK_BYTES as usize)
            .spawn(move || {
                if parent_loss(reader, &authority) {
                    // This entry point owns the worker root, never a host application.
                    std::process::exit(1);
                }
            })?;
        Ok(Self { armed })
    }

    /// Disarm before publishing a terminal frame: a parent may close its request
    /// pipe immediately after receiving that frame without changing worker success.
    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Release);
    }
}

impl Drop for WorkerParentGuard {
    fn drop(&mut self) {
        self.disarm();
    }
}

fn parent_loss(mut reader: impl Read, armed: &AtomicBool) -> bool {
    let mut byte = [0];
    loop {
        match reader.read(&mut byte) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            // No subsequent request or control frame is valid on this one-request pipe.
            _ => return armed.load(Ordering::Acquire),
        }
    }
}

#[cfg(test)]
mod tests;
