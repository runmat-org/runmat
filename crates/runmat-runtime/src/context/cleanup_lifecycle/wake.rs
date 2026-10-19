//! One host-owned safepoint dispatcher, woken by cross-thread GC root retirement.

use std::future::poll_fn;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Poll, Waker};

use std::sync::Mutex;

/// Notification only flips a bit and wakes the existing local-task driver.
/// It never inspects a root graph or executes a callback on the notifier lane.
pub(super) struct LifecycleWake {
    dirty: AtomicBool,
    closed: AtomicBool,
    waiter: Mutex<Option<Waker>>,
}

impl LifecycleWake {
    pub(super) fn new() -> Self {
        Self {
            dirty: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            waiter: Mutex::new(None),
        }
    }

    pub(super) fn notify(&self) {
        if self.closed.load(Ordering::Acquire) {
            return;
        }
        self.dirty.store(true, Ordering::Release);
        let waiter = self
            .waiter
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        if let Some(waiter) = waiter {
            waiter.wake();
        }
    }

    pub(super) fn close(&self) {
        self.closed.store(true, Ordering::Release);
        let waiter = self
            .waiter
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .take();
        if let Some(waiter) = waiter {
            waiter.wake();
        }
    }

    pub(super) async fn next(&self) -> bool {
        poll_fn(|context| {
            if self.closed.load(Ordering::Acquire) {
                return Poll::Ready(false);
            }
            if self.dirty.swap(false, Ordering::AcqRel) {
                return Poll::Ready(true);
            }
            *self
                .waiter
                .lock()
                .unwrap_or_else(|poison| poison.into_inner()) = Some(context.waker().clone());
            // A notification may race between the dirty check and Waker
            // publication; inspect again before returning Pending.
            if self.closed.load(Ordering::Acquire) {
                self.waiter
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .take();
                Poll::Ready(false)
            } else if self.dirty.swap(false, Ordering::AcqRel) {
                self.waiter
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .take();
                Poll::Ready(true)
            } else {
                Poll::Pending
            }
        })
        .await
    }
}
