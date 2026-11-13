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

/// Descriptive host selection, not authority minted by numeric coordinates.
/// The Runtime owner must validate every exact lease before and after a call.
#[derive(Clone, Debug)]
pub enum GraphicsCallbackRolePayload {
    Action {
        figure: super::GraphicsReceiverRef,
        scene: super::GraphicsSceneLease,
        axes: super::GraphicsReceiverRef,
    },
    Filter {
        figure: super::GraphicsReceiverRef,
        scene: super::GraphicsSceneLease,
        axes: super::GraphicsReceiverRef,
        target: super::GraphicsReceiverRef,
    },
}

impl GraphicsCallbackRolePayload {
    pub fn figure_and_scene(&self) -> (&super::GraphicsReceiverRef, &super::GraphicsSceneLease) {
        match self {
            Self::Action { figure, scene, .. } | Self::Filter { figure, scene, .. } => {
                (figure, scene)
            }
        }
    }

    pub fn matches_role(&self, kind: GraphicsCallbackKind) -> bool {
        matches!(
            (self, kind),
            (
                Self::Action { .. },
                GraphicsCallbackKind::PanActionPre
                    | GraphicsCallbackKind::PanActionPost
                    | GraphicsCallbackKind::ZoomActionPre
                    | GraphicsCallbackKind::ZoomActionPost
            ) | (
                Self::Filter { .. },
                GraphicsCallbackKind::PanButtonDownFilter
                    | GraphicsCallbackKind::ZoomButtonDownFilter
            )
        )
    }
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
