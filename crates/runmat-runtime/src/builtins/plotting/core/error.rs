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

/// Keep an owner's public wording and identifier while retaining the original
/// rendering failure for diagnostics and typed presentation classification.
pub(crate) fn with_render_cause(
    mut mapped: crate::RuntimeError,
    cause: crate::RuntimeError,
) -> crate::RuntimeError {
    if mapped.span.is_none() {
        mapped.span = cause.span;
    }
    mapped.gpu_gather_retry = cause.gpu_gather_retry;
    mapped.source = Some(Box::new(cause));
    mapped
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

    #[test]
    fn owner_wrapper_retains_identity_and_nested_typed_cause() {
        let cause = plotting_error_with_source(
            "plot",
            "native presentation failed",
            runmat_plot::PlotPresentationError::InteractiveBackend("setup failed".into()),
        );
        let cause = crate::build_runtime_error(cause.message().to_string())
            .with_source(cause)
            .with_span((11usize, 4usize).into())
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build();
        let owner = crate::build_runtime_error("owner-specific public wording")
            .with_builtin("boxplot")
            .with_identifier("RunMat:boxplot:InternalError")
            .build();
        let wrapped = with_render_cause(owner, cause);
        assert_eq!(wrapped.message(), "owner-specific public wording");
        assert_eq!(wrapped.identifier(), Some("RunMat:boxplot:InternalError"));
        assert_eq!(wrapped.span, Some((11usize, 4usize).into()));
        assert_eq!(wrapped.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert_eq!(
            presentation_failure_kind(&wrapped),
            Some(runmat_plot::PresentationFailureKind::InteractiveBackend)
        );
    }
}
