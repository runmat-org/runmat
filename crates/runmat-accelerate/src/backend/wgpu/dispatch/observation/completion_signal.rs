//! One bounded callback-to-completion signal owned by an observed submission.
//!
//! This replaces an opaque channel allocation at the provider boundary. The
//! existing WGPU progress owner still drives completion; this type is not a
//! scheduler or polling path.

#[cfg(target_arch = "wasm32")]
use std::future::poll_fn;
use std::sync::{Arc, Mutex};
#[cfg(target_arch = "wasm32")]
use std::task::Poll;
use std::task::Waker;

struct State<T> {
    value: Option<T>,
    sender_live: bool,
    waker: Option<Waker>,
}

struct Shared<T>(Mutex<State<T>>);

pub(super) struct Sender<T>(Option<Arc<Shared<T>>>);
pub(super) struct Receiver<T>(Arc<Shared<T>>);

pub(super) fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let shared = Arc::new(Shared(Mutex::new(State {
        value: None,
        sender_live: true,
        waker: None,
    })));
    (Sender(Some(shared.clone())), Receiver(shared))
}

pub(super) fn channel_host_upper_bytes<T>() -> Option<u64> {
    std::mem::size_of::<Shared<T>>()
        .checked_add(2 * std::mem::size_of::<usize>())?
        .checked_add(std::mem::align_of::<Shared<T>>())?
        .try_into()
        .ok()
}

impl<T> Sender<T> {
    pub(super) fn send(mut self, value: T) {
        let Some(shared) = self.0.take() else {
            return;
        };
        let wake = {
            let mut state = shared.0.lock().unwrap_or_else(|poison| poison.into_inner());
            state.value = Some(value);
            state.sender_live = false;
            state.waker.take()
        };
        if let Some(waker) = wake {
            waker.wake();
        }
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let Some(shared) = self.0.take() else {
            return;
        };
        let wake = {
            let mut state = shared.0.lock().unwrap_or_else(|poison| poison.into_inner());
            state.sender_live = false;
            state.waker.take()
        };
        if let Some(waker) = wake {
            waker.wake();
        }
    }
}

impl<T> Receiver<T> {
    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn take_after_progress(&mut self, progressed: bool) -> Option<T> {
        if !progressed {
            return None;
        }
        let mut state = self
            .0
             .0
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        state.value.take()
    }

    #[cfg(target_arch = "wasm32")]
    pub(super) async fn receive(&mut self) -> Option<T> {
        poll_fn(|context| {
            let mut next = Some(context.waker().clone());
            let (result, previous) = {
                let mut state = self
                    .0
                     .0
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner());
                if let Some(value) = state.value.take() {
                    (Poll::Ready(Some(value)), None)
                } else if !state.sender_live {
                    (Poll::Ready(None), None)
                } else {
                    (
                        Poll::Pending,
                        std::mem::replace(&mut state.waker, next.take()),
                    )
                }
            };
            // Replaced task wakers may run user code in Drop. Neither clone
            // nor destruction of those owners may happen under our lock.
            drop(previous);
            result
        })
        .await
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn known_layout_signal_distinguishes_delivery_from_sender_drop() {
        let (sender, mut receiver) = channel();
        sender.send(7_u32);
        assert_eq!(receiver.take_after_progress(true), Some(7));

        let (sender, mut receiver) = channel::<u32>();
        drop(sender);
        assert_eq!(receiver.take_after_progress(true), None);
    }
}
