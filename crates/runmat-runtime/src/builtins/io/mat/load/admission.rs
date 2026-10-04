//! One complete load receipt, retained through workspace or value publication.

use super::{LoadRequest, MatFsPath};
use crate::context::host_effect::{
    select_host_effect, HostEffectReceipt, HostEffectRequest, HostEffectWorkload, MatDecodeOutput,
    MatDecodeSelection,
};
use crate::context::RuntimeContext;
use crate::{BuiltinResult, RuntimeError};
use runmat_execution::{Digest, WholeWorkObservationOutcome};

pub(super) fn select(
    runtime: Option<&RuntimeContext>,
    source: &MatFsPath,
    request: &LoadRequest,
    output: MatDecodeOutput,
) -> BuiltinResult<Option<HostEffectReceipt>> {
    let selection = if request.variables.is_empty() && request.regex_tokens.is_empty() {
        MatDecodeSelection::All
    } else {
        MatDecodeSelection::Selected {
            names: &request.variables,
            patterns: &request.regex_tokens,
        }
    };
    let workload = HostEffectWorkload::MatDecode {
        source: source.path(),
        selection,
        output,
    };
    select_host_effect(
        runtime,
        HostEffectRequest {
            operation: "load",
            workload,
            domain_facts: Digest::sha256(workload.identity_bytes()),
            filesystem: source.provider(),
        },
    )
}

pub(super) fn finish<T>(receipt: Option<HostEffectReceipt>, result: &BuiltinResult<T>) {
    if let Some(receipt) = receipt {
        receipt.finish(match result {
            Ok(_) => WholeWorkObservationOutcome::Success,
            Err(error) if typed_cancelled(error) => WholeWorkObservationOutcome::Cancelled,
            Err(_) => WholeWorkObservationOutcome::OperationalFailure,
        });
    }
}

pub(super) fn finish_error(receipt: Option<HostEffectReceipt>, error: &RuntimeError) {
    if let Some(receipt) = receipt {
        receipt.finish(if typed_cancelled(error) {
            WholeWorkObservationOutcome::Cancelled
        } else {
            WholeWorkObservationOutcome::OperationalFailure
        });
    }
}

fn typed_cancelled(error: &RuntimeError) -> bool {
    let mut cause = error.source.as_deref();
    while let Some(current) = cause {
        if matches!(
            current.downcast_ref::<crate::execution::ExecutionServiceError>(),
            Some(crate::execution::ExecutionServiceError::Cancelled)
        ) {
            return true;
        }
        cause = current.source();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build_runtime_error;

    #[test]
    fn cancellation_is_classified_through_wrapped_typed_causes() {
        let inner = build_runtime_error("codec cancelled")
            .with_source(crate::execution::ExecutionServiceError::Cancelled)
            .build();
        let outer = build_runtime_error("load: wrapped codec error")
            .with_source(inner)
            .build();
        assert!(typed_cancelled(&outer));
        assert!(!typed_cancelled(
            &build_runtime_error("ordinary failure").build()
        ));
    }
}
