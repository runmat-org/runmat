pub(crate) fn plotting_error(builtin: &str, message: impl Into<String>) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(builtin)
        .build()
}

pub(crate) fn plotting_error_with_source(
    builtin: &str,
    message: impl Into<String>,
    source: impl crate::runtime_error::RuntimeErrorSourceBound,
) -> crate::RuntimeError {
    crate::build_runtime_error(message)
        .with_builtin(builtin)
        .with_source(source)
        .build()
}

/// Only platform-originated presentation failures may be ignored after a
/// figure has been mutated. Neither diagnostic text nor a caller's error
/// identifier establishes this policy.
pub(crate) fn presentation_failure_kind(
    error: &crate::RuntimeError,
) -> Option<runmat_plot::PresentationFailureKind> {
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(source) = current {
        if let Some(presentation) = source.downcast_ref::<runmat_plot::PlotPresentationError>() {
            return Some(presentation.kind());
        }
        current = source.source();
    }
    None
}

pub(crate) fn presentation_unavailable(error: &crate::RuntimeError) -> bool {
    matches!(
        presentation_failure_kind(error),
        Some(
            runmat_plot::PresentationFailureKind::Unavailable
                | runmat_plot::PresentationFailureKind::MainThreadRequired
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_typed_presentation_causes_are_ignored_even_when_nested() {
        let spoofed = plotting_error("plot", "plotting is unavailable on a non-main thread");
        assert!(!presentation_unavailable(&spoofed));

        let typed = plotting_error_with_source(
            "plot",
            "unchanged public diagnostic",
            runmat_plot::PlotPresentationError::MainThreadRequired,
        );
        let wrapped = plotting_error_with_source("plot", "outer", typed);
        assert_eq!(
            presentation_failure_kind(&wrapped),
            Some(runmat_plot::PresentationFailureKind::MainThreadRequired)
        );
        assert!(presentation_unavailable(&wrapped));
        let backend = plotting_error_with_source(
            "plot",
            "backend setup",
            runmat_plot::PlotPresentationError::InteractiveBackend("failed".into()),
        );
        assert!(!presentation_unavailable(&backend));
        assert_eq!(
            presentation_failure_kind(&backend),
            Some(runmat_plot::PresentationFailureKind::InteractiveBackend),
            "control plots retain their broader nonfatal setup policy"
        );
    }
}
