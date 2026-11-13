//! Cancellation is bound to the owned synchronous I/O thread, not another
//! session's stdin operation or a detached Tokio blocking-pool read.

use std::io::{self, Read};
use std::os::windows::io::{AsHandle, AsRawHandle};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use windows_sys::Win32::System::IO::CancelSynchronousIo;

pub(super) struct Reader {
    input: std::fs::File,
    cancelled: Arc<AtomicBool>,
}
pub(super) struct Cancel(Arc<AtomicBool>);

impl Reader {
    pub(super) fn stdin() -> io::Result<(Self, Cancel)> {
        let handle = std::io::stdin().as_handle().try_clone_to_owned()?;
        let cancelled = Arc::new(AtomicBool::new(false));
        Ok((
            Self {
                input: std::fs::File::from(handle),
                cancelled: Arc::clone(&cancelled),
            },
            Cancel(cancelled),
        ))
    }

    pub(super) fn read(mut self) -> io::Result<String> {
        let mut bytes = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            if self.cancelled.load(Ordering::Acquire) {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "REPL stdin cancelled",
                ));
            }
            match self.input.read(&mut chunk) {
                Ok(0) => {
                    return String::from_utf8(bytes)
                        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
                }
                Ok(count) => bytes.extend_from_slice(&chunk[..count]),
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }
    }
}

impl Cancel {
    pub(super) fn retire(&mut self, worker: &std::thread::JoinHandle<()>) {
        self.0.store(true, Ordering::Release);
        // Repeat across the check-before-ReadFile race; the next loop cannot
        // start another read after cancellation. Join proves actual retirement.
        while !worker.is_finished() {
            // The JoinHandle retains this exact native thread handle.
            unsafe {
                CancelSynchronousIo(worker.as_raw_handle());
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}
