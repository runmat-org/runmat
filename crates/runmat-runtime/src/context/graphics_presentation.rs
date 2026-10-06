//! Host-owned graphics presentation and callback-pump authority.
//!
//! A runtime context selects the service, not a figure registry. The service
//! owns the plot domain and decides which pending scene updates and callbacks
//! belong to a flush. Numeric figure handles are not session credentials.

use crate::RuntimeError;
use std::future::Future;
use std::pin::Pin;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsFlushRate {
    Immediate,
    LimitRate,
    /// Legacy `drawnow update`: skip while a renderer is busy, without the
    /// 50 ms limit-rate clock.
    WhenIdle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsCallbackPolicy {
    Process,
    Defer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphicsFlushRequest {
    pub rate: GraphicsFlushRate,
    pub callbacks: GraphicsCallbackPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsPresentation {
    Presented,
    DeferredByRate,
    DeferredByBusy,
    NoFigure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsCallbackReceipt {
    Processed,
    Deferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GraphicsFlushReceipt {
    pub presentation: GraphicsPresentation,
    pub callbacks: GraphicsCallbackReceipt,
}

/// Observable effect state of one presentation request. A host must switch
/// from `NotPublished` before it submits work that can reach a window or
/// canvas. It may report `Published` only after presentation commits. A lost
/// acknowledgement remains `Indeterminate`, never an invented no-effect. State
/// is monotonic: `NotPublished` → `Indeterminate` → `Published`, with no
/// regression after a host effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsFlushDisposition {
    NotPublished,
    Indeterminate,
    Published,
}

/// The invocation owner can inspect this disposition while a completion is
/// pending, including when cancellation or revocation drops the operation.
/// `abandon` atomically retires queued, unpublished work or reports that a
/// host dispatch already won the race. Returning `NotPublished` guarantees
/// the host cannot subsequently start the effect. This distinction cannot be
/// obtained by a separate disposition read followed by Drop.
pub trait RuntimeGraphicsFlushOperation:
    Future<Output = Result<GraphicsFlushReceipt, RuntimeError>>
{
    fn disposition(&self) -> GraphicsFlushDisposition;

    fn abandon(self: Pin<&mut Self>) -> GraphicsFlushDisposition;
}

/// Completes after the requested presentation and callback policy reaches a
/// terminal host disposition. A `NoFigure` presentation does not imply that
/// callbacks were processed. The service owns rate limiting, busy state, and
/// the callback queue; callers do not synthesize successful receipts.
///
/// The returned operation must begin no host effect before its first poll.
/// Drop retires pending work where possible; callers first use `abandon` to
/// atomically determine whether cancellation or host dispatch won.
pub trait RuntimeGraphicsPresentationService {
    fn flush(&self, request: GraphicsFlushRequest) -> Pin<Box<dyn RuntimeGraphicsFlushOperation>>;
}
