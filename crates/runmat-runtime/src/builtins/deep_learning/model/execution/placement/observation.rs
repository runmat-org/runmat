//! One genuine invocation's complete-route feedback, never a replay timer.

use runmat_execution::WholeWorkObservationOutcome;
use runmat_time::{duration_ns_saturating, Instant};
use std::error::Error;

use crate::context::{RuntimeContext, RuntimeWholeWorkSelection};

pub(in crate::builtins::deep_learning) struct PendingWholeWorkObservation {
    runtime: RuntimeContext,
    selected: RuntimeWholeWorkSelection,
    started: Instant,
}

impl PendingWholeWorkObservation {
    pub(in crate::builtins::deep_learning) fn start(
        runtime: &RuntimeContext,
        selected: RuntimeWholeWorkSelection,
    ) -> Self {
        Self {
            runtime: runtime.clone(),
            selected,
            started: Instant::now(),
        }
    }

    pub(in crate::builtins::deep_learning) fn complete(self) {
        self.observe(WholeWorkObservationOutcome::Success);
    }

    pub(in crate::builtins::deep_learning) fn operational_failure(self) {
        // The executor already classified this failure at its producing
        // operation. A later cancellation must not erase provider evidence.
        self.observe(WholeWorkObservationOutcome::OperationalFailure);
    }

    fn observe(self, outcome: WholeWorkObservationOutcome) {
        if let Some(service) = self.runtime.service_ports().placement() {
            if let Err(error) = service.observe_selected_whole_work(
                &self.selected,
                duration_ns_saturating(self.started.elapsed()),
                outcome,
            ) {
                tracing::warn!(target: "runmat_placement", "whole-network observation rejected: {error}");
            }
        }
        // The process-local exploration lease is released with `self` here.
    }
}

/// Download errors can also be caused by host allocation or presentation.
/// Only an owning provider execution source is negative provider evidence.
pub(in crate::builtins::deep_learning) fn is_provider_download_failure(
    error: &crate::RuntimeError,
) -> bool {
    let mut source = error
        .source
        .as_deref()
        .map(|source| source as &(dyn Error + 'static));
    while let Some(current) = source {
        if current.is::<runmat_accelerate_api::ProviderExecutionError>()
            || current.is::<crate::dispatcher::ProviderDownloadPayloadMismatch>()
        {
            return true;
        }
        source = current.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::is_provider_download_failure;

    #[test]
    fn only_an_owning_provider_source_counts_as_failed_download_evidence() {
        let provider = crate::runtime_error::with_provider_execution_source(
            crate::build_runtime_error("download failed").build(),
            runmat_accelerate_api::ProviderExecutionError::from_source(std::io::Error::other(
                "device failure",
            )),
        );
        assert!(is_provider_download_failure(&provider));
        let invalid_download = crate::build_runtime_error("malformed provider payload")
            .with_identifier("RunMat:gpu:ProviderPayloadMismatch")
            .with_source(crate::dispatcher::ProviderDownloadPayloadMismatch)
            .build();
        assert!(is_provider_download_failure(&invalid_download));
        let allocation = crate::build_runtime_error("host allocation failed").build();
        assert!(!is_provider_download_failure(&allocation));
        let invalid_handle = crate::build_runtime_error("provider owner unavailable")
            .with_identifier("RunMat:gather:ProviderUnavailable")
            .build();
        assert!(!is_provider_download_failure(&invalid_handle));
        let cancellation = crate::build_runtime_error("cancelled")
            .with_identifier("RunMat:Cancelled")
            .build();
        assert!(!is_provider_download_failure(&cancellation));
    }
}
