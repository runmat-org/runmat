//! Lifetime of a one-request isolated worker, independent of its async executor.
use super::frame::{read_payload_blocking, FrameLimits};
use crate::ProcessHostResult;
use std::io::Read;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

// This thread frames and decodes typed control messages through host observers.
// Reserve the standard Rust worker-stack size: a 64 KiB I/O-only stack overflows
// on a valid debug-profile bootstrap. Admission quotes use this same constant.
pub const WORKER_PARENT_MONITOR_STACK_BYTES: u64 = 2 * 1024 * 1024;

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

/// Frames the parent writes after the request (cancellation, collective replies), in order.
/// An input error after the link is disarmed arrives as the final item.
pub type ParentFrames = tokio::sync::mpsc::UnboundedReceiver<ProcessHostResult<Vec<u8>>>;
/// A private protocol owner may consume its own bounded controls on the
/// existing reader thread; unrelated frames continue to the original receiver.
pub type ParentFrameObserver = Arc<dyn Fn(&[u8]) -> bool + Send + Sync>;

/// For a worker whose request pipe also carries later control frames. A dedicated thread owns
/// the pipe and forwards every frame; EOF or an input error while armed is parent loss and ends
/// the worker root at once, whether or not the executor is polling (a solver or a source loop
/// need not yield). Disarm before publishing a terminal frame, as for [`WorkerParentGuard`].
pub struct WorkerParentLink {
    armed: Arc<AtomicBool>,
}

impl WorkerParentLink {
    pub fn start(
        reader: impl Read + Send + 'static,
        limits: FrameLimits,
    ) -> ProcessHostResult<(Self, ParentFrames)> {
        Self::start_with_observer(reader, limits, None)
    }
    pub fn start_with_observer(
        reader: impl Read + Send + 'static,
        limits: FrameLimits,
        observer: Option<ParentFrameObserver>,
    ) -> ProcessHostResult<(Self, ParentFrames)> {
        limits.validate()?;
        let armed = Arc::new(AtomicBool::new(true));
        let authority = Arc::clone(&armed);
        let (frames, receiver) = tokio::sync::mpsc::unbounded_channel();
        std::thread::Builder::new()
            .name("runmat-worker-parent".into())
            .stack_size(WORKER_PARENT_MONITOR_STACK_BYTES as usize)
            .spawn(move || {
                if forward_frames(reader, limits, &authority, &frames, observer.as_ref()) {
                    // This entry point owns the worker root, never a host application.
                    std::process::exit(1);
                }
            })?;
        Ok((Self { armed }, receiver))
    }

    pub fn disarm(&self) {
        self.armed.store(false, Ordering::Release);
    }
}

impl Drop for WorkerParentLink {
    fn drop(&mut self) {
        self.disarm();
    }
}

impl WorkerParentLink {
    /// One queued ordinary control; observer replies bypass that slot even while
    /// synchronous numerical work occupies the executor. Protocol flooding fails
    /// closed rather than blocking the reader or retaining unbounded controls.
    pub fn start_bounded_with_observer(
        reader: impl Read + Send + 'static,
        limits: FrameLimits,
        observer: Option<ParentFrameObserver>,
    ) -> ProcessHostResult<(
        Self,
        tokio::sync::mpsc::Receiver<ProcessHostResult<Vec<u8>>>,
    )> {
        limits.validate()?;
        let armed = Arc::new(AtomicBool::new(true));
        let authority = armed.clone();
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        std::thread::Builder::new()
            .name("runmat-worker-parent".into())
            .stack_size(WORKER_PARENT_MONITOR_STACK_BYTES as usize)
            .spawn(move || {
                if forward_frames_with_send(
                    reader,
                    limits,
                    &authority,
                    observer.as_ref(),
                    |frame| sender.try_send(frame).is_ok(),
                ) {
                    std::process::exit(1);
                }
            })?;
        Ok((Self { armed }, receiver))
    }
}

/// Forwards frames until the pipe ends; true when it ended while armed (parent loss).
fn forward_frames(
    reader: impl Read,
    limits: FrameLimits,
    armed: &AtomicBool,
    frames: &tokio::sync::mpsc::UnboundedSender<ProcessHostResult<Vec<u8>>>,
    observer: Option<&ParentFrameObserver>,
) -> bool {
    forward_frames_with_send(reader, limits, armed, observer, |frame| {
        // Existing unbounded consumers may finish before parent loss; keep
        // watching their pipe until the original link is disarmed.
        let _ = frames.send(frame);
        true
    })
}

fn forward_frames_with_send(
    mut reader: impl Read,
    limits: FrameLimits,
    armed: &AtomicBool,
    observer: Option<&ParentFrameObserver>,
    send: impl Fn(ProcessHostResult<Vec<u8>>) -> bool,
) -> bool {
    loop {
        match read_payload_blocking(&mut reader, limits) {
            // A receiver that has gone away still leaves the pipe to watch for loss.
            Ok(frame) => {
                if !observer.is_some_and(|observer| observer(&frame)) && !send(Ok(frame)) {
                    return armed.load(Ordering::Acquire);
                }
            }
            Err(error) => {
                if armed.load(Ordering::Acquire) {
                    return true;
                }
                let _ = send(Err(error));
                return false;
            }
        }
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
