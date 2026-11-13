//! Readiness plus a private cancellation descriptor; no stdin flag mutation.

use std::io::{self, Read};
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::net::UnixStream;

pub(super) struct Reader {
    input: std::fs::File,
    cancelled: UnixStream,
}
pub(super) struct Cancel(Option<UnixStream>);

impl Reader {
    pub(super) fn stdin() -> io::Result<(Self, Cancel)> {
        let input = std::io::stdin().as_fd().try_clone_to_owned()?;
        Self::new(std::fs::File::from(input))
    }

    pub(super) fn new(input: std::fs::File) -> io::Result<(Self, Cancel)> {
        let (cancelled, cancel) = UnixStream::pair()?;
        Ok((Self { input, cancelled }, Cancel(Some(cancel))))
    }

    pub(super) fn read(mut self) -> io::Result<String> {
        let mut bytes = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            let mut ready = [
                libc::pollfd {
                    fd: self.cancelled.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
                libc::pollfd {
                    fd: self.input.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                },
            ];
            // Both descriptors remain owned for this call. This CLI reader is
            // stdin's sole consumer; after readiness each read is bounded.
            let result = unsafe { libc::poll(ready.as_mut_ptr(), 2, -1) };
            if result < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error);
            }
            if ready[0].revents != 0 {
                return Err(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "REPL stdin cancelled",
                ));
            }
            if ready[1].revents & libc::POLLNVAL != 0 {
                return Err(io::Error::other("REPL stdin descriptor retired"));
            }
            if ready[1].revents == 0 {
                continue;
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
    pub(super) fn retire(&mut self, _: &std::thread::JoinHandle<()>) {
        // Closing the sole cancellation peer wakes POLLHUP without a fallible
        // shutdown operation or mutation of stdin's shared file-status flags.
        drop(self.0.take());
    }
}
