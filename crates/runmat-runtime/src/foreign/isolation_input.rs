//! Shared input ownership for native isolation hosts.
//!
//! A dedicated blocking reader delivers validated driver records to one
//! mailbox. The host polls that mailbox alongside its existing local task
//! supervisor while idle; synchronous callback reentry consumes from the
//! same FIFO without holding a channel RefCell across an await.

use std::io::Read;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll};

use futures::task::AtomicWaker;

pub(super) struct InputMailbox<T> {
    state: Mutex<InputState<T>>,
    changed: Condvar,
    wake: AtomicWaker,
}

struct InputState<T> {
    queued: Option<Result<T, String>>,
    closed: bool,
    reader_done: bool,
}

/// Process-host lifetime owner. Close wakes both a blocked synchronous
/// callback reader and the bounded producer without requiring either to
/// relinquish ownership first.
pub(super) struct InputMailboxClose<T>(Arc<InputMailbox<T>>);

impl<T> InputMailboxClose<T> {
    pub(super) fn new(mailbox: Arc<InputMailbox<T>>) -> Self {
        Self(mailbox)
    }
}

impl<T> Drop for InputMailboxClose<T> {
    fn drop(&mut self) {
        self.0.close();
    }
}

impl<T> InputMailbox<T> {
    pub(super) fn read_blocking(&self) -> Result<T, String> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| "isolation input mailbox is poisoned".to_string())?;
        loop {
            if state.closed {
                return Err("isolation input mailbox is closed".to_string());
            }
            if let Some(message) = state.queued.take() {
                self.changed.notify_all();
                return message;
            }
            if state.reader_done {
                return Err("isolation input reader closed".to_string());
            }
            state = self
                .changed
                .wait(state)
                .map_err(|_| "isolation input mailbox is poisoned".to_string())?;
        }
    }

    pub(super) async fn next(&self) -> Result<T, String> {
        futures::future::poll_fn(|context: &mut Context<'_>| {
            self.wake.register(context.waker());
            let mut state = match self.state.lock() {
                Ok(state) => state,
                Err(_) => {
                    return Poll::Ready(Err("isolation input mailbox is poisoned".to_string()))
                }
            };
            if state.closed {
                Poll::Ready(Err("isolation input mailbox is closed".to_string()))
            } else if let Some(message) = state.queued.take() {
                self.changed.notify_all();
                Poll::Ready(message)
            } else if state.reader_done {
                Poll::Ready(Err("isolation input reader closed".to_string()))
            } else {
                Poll::Pending
            }
        })
        .await
    }

    pub(super) fn close(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.closed = true;
            state.queued.take();
        }
        self.changed.notify_all();
        self.wake.wake();
    }

    fn send_blocking(&self, message: Result<T, String>) -> bool {
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        while state.queued.is_some() && !state.closed {
            let Ok(next) = self.changed.wait(state) else {
                return false;
            };
            state = next;
        }
        if state.closed {
            return false;
        }
        state.queued = Some(message);
        drop(state);
        self.changed.notify_all();
        self.wake.wake();
        true
    }

    fn finish_reader(&self) {
        if let Ok(mut state) = self.state.lock() {
            state.reader_done = true;
        }
        self.changed.notify_all();
        self.wake.wake();
    }
}

impl<T> Drop for InputMailbox<T> {
    fn drop(&mut self) {
        self.close();
    }
}

// One queued frame plus the reader's current decoded frame bounds payload
// retention by the existing per-frame protocol limit, even under a stalled
// callback or a maliciously fast driver.
const INPUT_MAILBOX_CAPACITY: usize = 1;

pub(super) fn spawn_input_reader<R, T>(
    mut reader: R,
    thread_name: &'static str,
    mut decode: impl FnMut(&mut R) -> Result<T, String> + Send + 'static,
) -> Result<Arc<InputMailbox<T>>, String>
where
    R: Read + Send + 'static,
    T: Send + 'static,
{
    debug_assert_eq!(INPUT_MAILBOX_CAPACITY, 1);
    let mailbox = Arc::new(InputMailbox {
        state: Mutex::new(InputState {
            queued: None,
            closed: false,
            reader_done: false,
        }),
        changed: Condvar::new(),
        wake: AtomicWaker::new(),
    });
    let reader_mailbox = Arc::downgrade(&mailbox);
    std::thread::Builder::new()
        .name(thread_name.into())
        .spawn(move || {
            loop {
                let message = decode(&mut reader);
                let terminal = message.is_err();
                let Some(mailbox) = reader_mailbox.upgrade() else {
                    break;
                };
                if !mailbox.send_blocking(message) {
                    break;
                }
                if terminal {
                    break;
                }
                drop(mailbox);
            }
            if let Some(mailbox) = reader_mailbox.upgrade() {
                mailbox.finish_reader();
            }
        })
        .map_err(|error| format!("could not start isolation input reader: {error}"))?;
    Ok(mailbox)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[test]
    fn idle_async_and_callback_blocking_reads_share_one_ordered_mailbox() {
        let input = std::io::Cursor::new(b"ab".to_vec());
        let mailbox = spawn_input_reader(input, "runmat-isolation-input-test", |reader| {
            let mut byte = [0u8; 1];
            reader
                .read_exact(&mut byte)
                .map_err(|error| error.to_string())?;
            Ok(byte[0])
        })
        .expect("reader thread");
        assert_eq!(futures::executor::block_on(mailbox.next()), Ok(b'a'));
        assert_eq!(mailbox.read_blocking(), Ok(b'b'));
        assert!(futures::executor::block_on(mailbox.next()).is_err());
    }

    #[test]
    fn stalled_consumer_backpressures_reader_and_close_retires_it() {
        struct ExitFlag(Arc<AtomicBool>);
        impl Drop for ExitFlag {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let decoded = Arc::new(AtomicUsize::new(0));
        let finished = Arc::new(AtomicBool::new(false));
        let decoded_in_reader = Arc::clone(&decoded);
        let exit = ExitFlag(Arc::clone(&finished));
        let mailbox = spawn_input_reader(
            std::io::Cursor::new(vec![b'x'; 128]),
            "runmat-isolation-backpressure-test",
            move |reader| {
                let _keep_exit_flag = &exit;
                let mut byte = [0u8; 1];
                reader
                    .read_exact(&mut byte)
                    .map_err(|error| error.to_string())?;
                decoded_in_reader.fetch_add(1, Ordering::Relaxed);
                Ok(byte[0])
            },
        )
        .expect("reader thread");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(
            decoded.load(Ordering::Relaxed) <= INPUT_MAILBOX_CAPACITY + 1,
            "reader decoded beyond one queued and one pending frame"
        );
        mailbox.close();
        for _ in 0..100 {
            if finished.load(Ordering::Acquire) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(finished.load(Ordering::Acquire), "reader did not retire");
    }

    #[test]
    fn close_unblocks_callback_waiting_for_next_frame() {
        let mailbox = Arc::new(InputMailbox::<u8> {
            state: Mutex::new(InputState {
                queued: None,
                closed: false,
                reader_done: false,
            }),
            changed: Condvar::new(),
            wake: AtomicWaker::new(),
        });
        let (ready, started) = std::sync::mpsc::sync_channel(1);
        let (finished, result) = std::sync::mpsc::sync_channel(1);
        let reader = Arc::clone(&mailbox);
        let thread = std::thread::spawn(move || {
            ready.send(()).expect("reader started");
            finished
                .send(reader.read_blocking())
                .expect("reader outcome");
        });
        started.recv().expect("reader ready");
        mailbox.close();
        assert!(result
            .recv_timeout(std::time::Duration::from_secs(1))
            .expect("close woke callback")
            .is_err());
        thread.join().expect("reader joined");
    }
}
