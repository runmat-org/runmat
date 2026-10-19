//! One finite off-lane stdin reader. Cancelling its future wakes/cancels and
//! joins that exact reader before the session execution lane can retire.

use futures::channel::oneshot;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

#[cfg(unix)]
#[path = "piped_input/unix.rs"]
mod platform;
#[cfg(windows)]
#[path = "piped_input/windows.rs"]
mod platform;

static STDIN_OWNED: AtomicBool = AtomicBool::new(false);

struct StdinLease;
impl StdinLease {
    fn acquire() -> io::Result<Self> {
        STDIN_OWNED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| Self)
            .map_err(|_| io::Error::other("CLI stdin already has a reader"))
    }
}
impl Drop for StdinLease {
    fn drop(&mut self) {
        STDIN_OWNED.store(false, Ordering::Release);
    }
}

pub(crate) struct PipedInput {
    receiver: oneshot::Receiver<io::Result<String>>,
    worker: Option<std::thread::JoinHandle<()>>,
    cancel: platform::Cancel,
}

impl PipedInput {
    /// CLI-only exclusive stdin ownership; not a generic Runtime stream policy.
    pub(crate) fn stdin() -> io::Result<Self> {
        let lease = StdinLease::acquire()?;
        let (reader, cancel) = platform::Reader::stdin()?;
        Self::start(reader, cancel, Some(lease))
    }

    fn start(
        reader: platform::Reader,
        cancel: platform::Cancel,
        lease: Option<StdinLease>,
    ) -> io::Result<Self> {
        let (sender, receiver) = oneshot::channel();
        let worker = std::thread::Builder::new()
            .name("runmat-stdin-input".into())
            .spawn(move || {
                let _lease = lease;
                let result = reader.read();
                let _ = sender.send(result);
            })?;
        Ok(Self {
            receiver,
            worker: Some(worker),
            cancel,
        })
    }

    fn retire(&mut self) -> io::Result<()> {
        if let Some(worker) = self.worker.as_ref() {
            self.cancel.retire(worker);
        }
        if let Some(worker) = self.worker.take() {
            worker
                .join()
                .map_err(|_| io::Error::other("CLI stdin reader panicked"))?;
        }
        Ok(())
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::fd::{AsRawFd, OwnedFd};
    use std::os::unix::net::UnixStream;

    #[test]
    fn pipe_input_finishes_at_eof_and_pending_drop_joins_without_stdin_flag_changes() {
        for complete in [true, false] {
            let (input, mut writer) = UnixStream::pair().unwrap();
            // This retained descriptor also proves the open-description flags
            // are unchanged while the reader owns its ordinary duplicate.
            let retained = input.try_clone().unwrap();
            let flags = unsafe { libc::fcntl(retained.as_raw_fd(), libc::F_GETFL) };
            assert!(flags >= 0);
            let owned: OwnedFd = input.into();
            let (reader, cancel) = platform::Reader::new(std::fs::File::from(owned)).unwrap();
            let mut future = Box::pin(PipedInput::start(reader, cancel, None).unwrap());
            if complete {
                writer.write_all("answer = 42;\n".as_bytes()).unwrap();
                drop(writer);
                assert_eq!(
                    futures::executor::block_on(future).unwrap(),
                    "answer = 42;\n"
                );
            } else {
                let waker = futures::task::noop_waker();
                assert!(future
                    .as_mut()
                    .poll(&mut Context::from_waker(&waker))
                    .is_pending());
                let (done, completed) = std::sync::mpsc::sync_channel(1);
                let retire = std::thread::spawn(move || {
                    drop(future);
                    done.send(()).unwrap();
                });
                completed
                    .recv_timeout(std::time::Duration::from_secs(2))
                    .unwrap();
                retire.join().unwrap();
                drop(writer);
            }
            assert_eq!(
                unsafe { libc::fcntl(retained.as_raw_fd(), libc::F_GETFL) },
                flags
            );
        }
    }
}

impl Future for PipedInput {
    type Output = io::Result<String>;
    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let result = match Pin::new(&mut self.receiver).poll(context) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Ok(result)) => result,
            Poll::Ready(Err(_)) => Err(io::Error::other("CLI stdin reader returned no result")),
        };
        let retired = self.retire();
        Poll::Ready(match (result, retired) {
            (Err(primary), Err(cleanup)) => {
                eprintln!("CLI stdin retirement failed: {cleanup}");
                Err(primary)
            }
            (Err(primary), _) => Err(primary),
            (Ok(_), Err(cleanup)) => Err(cleanup),
            (Ok(value), Ok(())) => Ok(value),
        })
    }
}

impl Drop for PipedInput {
    fn drop(&mut self) {
        if let Err(error) = self.retire() {
            eprintln!("CLI stdin retirement failed: {error}");
        }
    }
}
