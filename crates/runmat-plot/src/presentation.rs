//! Platform-owned reasons why interactive figure presentation failed.

use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresentationFailureKind {
    Unavailable,
    MainThreadRequired,
    InteractiveBackend,
}

#[derive(Debug, Error)]
pub enum PlotPresentationError {
    #[error("{0}")]
    Unavailable(&'static str),
    #[error("Interactive plotting is unavailable on macOS when called from a non-main thread. Launch RunMat from the main thread, or use file export APIs for headless rendering.")]
    MainThreadRequired,
    #[error("{0}")]
    InteractiveBackend(String),
}

impl PlotPresentationError {
    pub const fn kind(&self) -> PresentationFailureKind {
        match self {
            Self::Unavailable(_) => PresentationFailureKind::Unavailable,
            Self::MainThreadRequired => PresentationFailureKind::MainThreadRequired,
            Self::InteractiveBackend(_) => PresentationFailureKind::InteractiveBackend,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_expose_stable_semantics_separate_from_display_text() {
        assert_eq!(
            PlotPresentationError::Unavailable("presentation disabled").kind(),
            PresentationFailureKind::Unavailable
        );
        assert_eq!(
            PlotPresentationError::MainThreadRequired.kind(),
            PresentationFailureKind::MainThreadRequired
        );
        assert_eq!(
            PlotPresentationError::InteractiveBackend("backend failed".into()).kind(),
            PresentationFailureKind::InteractiveBackend
        );
    }

    #[cfg(not(feature = "gui"))]
    #[test]
    fn interactive_entry_reports_typed_no_gui_unavailability() {
        let error = crate::present_interactive_with_handle(1, crate::plots::Figure::new())
            .expect_err("no GUI build cannot present interactively");
        assert_eq!(error.kind(), PresentationFailureKind::Unavailable);
    }
}
