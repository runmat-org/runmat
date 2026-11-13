//! One native window's cancellable frame command and post-present receipt.
//!
//! The window event loop owns the actual renderer. This ticket never runs a
//! blocking window loop or treats command delivery as a presented frame.

use crate::plots::Figure;
use futures::channel::oneshot;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use winit::event_loop::EventLoopProxy;
use winit::window::WindowId;

const QUEUED: u8 = 0;
const RETIRED: u8 = 1;
const DISPATCHED: u8 = 2;
const PRESENTED: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDisposition {
    NotPublished,
    Indeterminate,
    Published,
}

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("the plot window is no longer accepting frames")]
    WindowClosed,
    #[error("plot frame preparation failed: {0}")]
    Prepare(String),
    #[error("plot frame rendering failed: {0}")]
    Render(String),
}

#[derive(Default)]
pub(super) struct FrameState(AtomicU8);

impl FrameState {
    fn begin(&self) -> bool {
        self.0
            .compare_exchange(QUEUED, DISPATCHED, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    fn presented(&self) {
        let prior = self.0.swap(PRESENTED, Ordering::AcqRel);
        debug_assert_eq!(prior, DISPATCHED);
    }

    fn abandon(&self) -> FrameDisposition {
        let state =
            match self
                .0
                .compare_exchange(QUEUED, RETIRED, Ordering::AcqRel, Ordering::Acquire)
            {
                Ok(_) => RETIRED,
                Err(state) => state,
            };
        match state {
            QUEUED | RETIRED => FrameDisposition::NotPublished,
            DISPATCHED => FrameDisposition::Indeterminate,
            PRESENTED => FrameDisposition::Published,
            _ => unreachable!("closed frame state"),
        }
    }

    fn disposition(&self) -> FrameDisposition {
        match self.0.load(Ordering::Acquire) {
            QUEUED | RETIRED => FrameDisposition::NotPublished,
            DISPATCHED => FrameDisposition::Indeterminate,
            PRESENTED => FrameDisposition::Published,
            _ => unreachable!("closed frame state"),
        }
    }
}

pub(super) struct FrameCommand {
    target: FrameTarget,
    figure: Option<Box<Figure>>,
    result: Option<oneshot::Sender<Result<(), FrameError>>>,
    state: Arc<FrameState>,
}

enum FrameTarget {
    Window {
        id: WindowId,
        lifetime: Arc<std::sync::atomic::AtomicBool>,
    },
    #[cfg(test)]
    Synthetic,
}

impl FrameCommand {
    pub(super) fn window_id(&self) -> Option<WindowId> {
        match &self.target {
            FrameTarget::Window { id, .. } => Some(*id),
            #[cfg(test)]
            FrameTarget::Synthetic => None,
        }
    }

    pub(super) fn matches_window_lifetime(
        &self,
        current: &Arc<std::sync::atomic::AtomicBool>,
    ) -> bool {
        match &self.target {
            FrameTarget::Window { lifetime, .. } => live_window_lifetime(lifetime, current),
            #[cfg(test)]
            FrameTarget::Synthetic => false,
        }
    }

    pub(super) fn begin(mut self) -> Option<InFlightFrame> {
        if !self.state.begin() {
            return None;
        }
        Some(InFlightFrame {
            figure: self.figure.take(),
            result: self.result.take().expect("queued command owns receipt"),
            state: Arc::clone(&self.state),
        })
    }
}

fn live_window_lifetime(
    issued: &Arc<std::sync::atomic::AtomicBool>,
    current: &Arc<std::sync::atomic::AtomicBool>,
) -> bool {
    Arc::ptr_eq(issued, current) && !issued.load(Ordering::Acquire)
}

impl Drop for FrameCommand {
    fn drop(&mut self) {
        self.state.abandon();
    }
}

pub(super) struct InFlightFrame {
    figure: Option<Box<Figure>>,
    result: oneshot::Sender<Result<(), FrameError>>,
    state: Arc<FrameState>,
}

impl InFlightFrame {
    pub(super) fn take_figure(&mut self) -> Figure {
        *self.figure.take().expect("dispatched frame owns figure")
    }

    pub(super) fn complete(self, result: Result<(), FrameError>) {
        if result.is_ok() {
            self.state.presented();
        }
        let _ = self.result.send(result);
    }
}

/// A producer may submit only to this already-created window event loop.
#[derive(Clone)]
pub struct FramePort {
    sender: Arc<dyn FrameCommandSender>,
    window_id: WindowId,
    close_signal: Arc<std::sync::atomic::AtomicBool>,
}

// Keep the native event-loop transport behind its typed sending boundary.
// Ports and receipts expose frame authority, not the renderer's type graph.
trait FrameCommandSender: Send + Sync {
    fn send(&self, command: FrameCommand) -> Result<(), FrameError>;
}

struct NativeFrameCommandSender(EventLoopProxy<FrameCommand>);

impl FrameCommandSender for NativeFrameCommandSender {
    fn send(&self, command: FrameCommand) -> Result<(), FrameError> {
        self.0
            .send_event(command)
            .map_err(|_| FrameError::WindowClosed)
    }
}

impl FramePort {
    pub(super) fn new(
        proxy: EventLoopProxy<FrameCommand>,
        window_id: WindowId,
        close_signal: Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        Self {
            sender: Arc::new(NativeFrameCommandSender(proxy)),
            window_id,
            close_signal,
        }
    }

    /// Retire this exact window, never another attachment's handle route.
    pub fn close(&self) {
        self.close_signal.store(true, Ordering::Release);
    }

    pub fn present(&self, figure: Figure) -> FrameTicket {
        let (result, receiver) = oneshot::channel();
        let state = Arc::new(FrameState::default());
        FrameTicket {
            sender: Arc::clone(&self.sender),
            command: Some(FrameCommand {
                target: FrameTarget::Window {
                    id: self.window_id,
                    lifetime: Arc::clone(&self.close_signal),
                },
                figure: Some(Box::new(figure)),
                result: Some(result),
                state: Arc::clone(&state),
            }),
            receiver,
            state,
        }
    }
}

/// Lazy frame request; dropping before host dispatch prevents a frame effect.
pub struct FrameTicket {
    sender: Arc<dyn FrameCommandSender>,
    command: Option<FrameCommand>,
    receiver: oneshot::Receiver<Result<(), FrameError>>,
    state: Arc<FrameState>,
}

impl FrameTicket {
    pub fn disposition(&self) -> FrameDisposition {
        self.state.disposition()
    }

    pub fn abandon(&self) -> FrameDisposition {
        self.state.abandon()
    }
}

impl Future for FrameTicket {
    type Output = Result<(), FrameError>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(command) = self.command.take() {
            if let Err(error) = self.sender.send(command) {
                self.state.abandon();
                return Poll::Ready(Err(error));
            }
        }
        match Pin::new(&mut self.receiver).poll(context) {
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(_)) => Poll::Ready(Err(FrameError::WindowClosed)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for FrameTicket {
    fn drop(&mut self) {
        self.state.abandon();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn abandonment_and_dispatch_have_one_atomic_winner() {
        for _ in 0..100 {
            let state = Arc::new(FrameState::default());
            let barrier = Arc::new(Barrier::new(2));
            let other_state = Arc::clone(&state);
            let other_barrier = Arc::clone(&barrier);
            let host = std::thread::spawn(move || {
                other_barrier.wait();
                other_state.begin()
            });
            barrier.wait();
            let abandoned = state.abandon();
            let dispatched = host.join().expect("host race joined");
            assert_eq!(abandoned == FrameDisposition::NotPublished, !dispatched);
            assert_eq!(state.disposition(), abandoned);
        }
    }

    #[test]
    fn presented_receipt_survives_late_abandonment() {
        let state = FrameState::default();
        assert!(state.begin());
        state.presented();
        assert_eq!(state.abandon(), FrameDisposition::Published);
        assert_eq!(state.abandon(), FrameDisposition::Published);
    }

    #[test]
    fn queued_drop_never_dispatches_and_post_present_receipt_is_terminal() {
        let lifetime = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let same_window = Arc::clone(&lifetime);
        assert!(live_window_lifetime(&lifetime, &same_window));
        let next_window = Arc::new(std::sync::atomic::AtomicBool::new(false));
        assert!(!live_window_lifetime(&lifetime, &next_window));
        lifetime.store(true, Ordering::Release);
        assert!(!live_window_lifetime(&lifetime, &same_window));
        let (sender, receiver) = oneshot::channel();
        let state = Arc::new(FrameState::default());
        let command = FrameCommand {
            target: FrameTarget::Synthetic,
            figure: Some(Box::new(Figure::new())),
            result: Some(sender),
            state: Arc::clone(&state),
        };
        drop(command);
        assert!(!state.begin());
        assert_eq!(state.disposition(), FrameDisposition::NotPublished);
        assert!(futures::executor::block_on(receiver).is_err());

        let (sender, receiver) = oneshot::channel();
        let state = Arc::new(FrameState::default());
        let command = FrameCommand {
            target: FrameTarget::Synthetic,
            figure: Some(Box::new(Figure::new())),
            result: Some(sender),
            state: Arc::clone(&state),
        };
        let mut frame = command.begin().expect("host won dispatch admission");
        let _figure = frame.take_figure();
        assert_eq!(state.abandon(), FrameDisposition::Indeterminate);
        frame.complete(Ok(()));
        assert!(futures::executor::block_on(receiver).unwrap().is_ok());
        assert_eq!(state.abandon(), FrameDisposition::Published);
    }
}
