//! Semantic graphics callback operation and result types shared by the host
//! presentation and the originating session's callable adapter.
//!
//! These values carry no callable, GC root, runtime context, or session
//! authority. The live Figure/Axes lease and selected session are checked at
//! the respective plotting and runtime owners before any callback executes.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphicsCallbackKind {
    DataCursorUpdate,
    PanButtonDownFilter,
    PanActionPre,
    PanActionPost,
    ZoomButtonDownFilter,
    ZoomActionPre,
    ZoomActionPost,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphicsCallbackResult {
    Completed,
    Suppress(bool),
    DataTipText(Vec<String>),
}

/// Host-visible result of an admitted request. Detailed RuntimeError and
/// callable roots remain on the originating session lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphicsCallbackHostFailure {
    CallbackFailed,
    InvalidResult,
    Retired,
    /// The selected callback may already have produced effects when its
    /// session retired. The host must not replay the gesture.
    Indeterminate,
}

/// Whether a host gesture may have started its callback. The native event
/// lane may drop a queued ticket only when retirement wins NotPublished.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphicsCallbackDisposition {
    NotPublished,
    Indeterminate,
    Published,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphicsCallbackIngressFailure {
    MismatchedScope,
    Retired,
    Full,
    Closed,
}
