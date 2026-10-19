//! Typed applicability versus closed resource refusal, independent of receipts.
use crate::{build_runtime_error, BuiltinResult};
use runmat_accelerate_api::{ProviderFeasibility, ProviderRejectionCode, ProviderResourceEstimate};
use runmat_execution::{ProviderResourceSnapshot, ResourceAdmissionError};

pub(super) fn finite_budget(snapshot: &ProviderResourceSnapshot, host_budget: Option<u64>) -> bool {
    host_budget.is_some()
        || snapshot.capacity_bytes.is_some()
        || snapshot.scratch_available_bytes.is_some()
        || snapshot.queue_limit.is_some()
}

pub(super) fn require_queue_bound(
    snapshot: &ProviderResourceSnapshot,
    bound: Option<u32>,
) -> BuiltinResult<()> {
    if bound.is_none() && snapshot.queue_limit.is_some() {
        return Err(build_runtime_error(
            "provider operation has no physical queue demand under a finite queue budget",
        )
        .with_identifier("RunMat:Placement:InsufficientResourceEvidence")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    Ok(())
}

pub(super) fn estimate(
    feasibility: ProviderFeasibility,
    snapshot: &ProviderResourceSnapshot,
    host_budget: Option<u64>,
) -> BuiltinResult<Option<ProviderResourceEstimate>> {
    use ProviderRejectionCode as Code;
    match feasibility {
        ProviderFeasibility::Supported { estimate } => {
            validate_live(snapshot)?;
            Ok(Some(estimate))
        }
        ProviderFeasibility::Rejected { rejection } => {
            let identifier = match rejection.code {
                Code::UnsupportedOperation
                | Code::UnsupportedElementType
                | Code::UnsupportedStorage
                | Code::UnsupportedLayout
                | Code::UnsupportedRank
                | Code::InvalidShape => {
                    // None means no placement receipt, not no provider attempt:
                    // legacy callers still invoke their operation-owned hook.
                    validate_live(snapshot)?;
                    if !finite_budget(snapshot, host_budget) {
                        return Ok(None);
                    }
                    "RunMat:Placement:InsufficientResourceEvidence"
                }
                Code::ResourceLimit | Code::ProviderUnavailable => {
                    "RunMat:Placement:NoLegalCandidate"
                }
                Code::UnknownResourceEstimate => {
                    validate_live(snapshot)?;
                    if !finite_budget(snapshot, host_budget) {
                        return Ok(None);
                    }
                    "RunMat:Placement:InsufficientResourceEvidence"
                }
            };
            Err(
                build_runtime_error(format!("provider operation admission refused: {rejection}"))
                    .with_identifier(identifier)
                    .with_source(rejection)
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build(),
            )
        }
    }
}

pub(super) fn validate_live(snapshot: &ProviderResourceSnapshot) -> BuiltinResult<()> {
    snapshot
        .validate_typed()
        .map_err(|cause| super::resource_error(ResourceAdmissionError::InvalidSnapshot(cause)))?;
    if snapshot.lost {
        return Err(super::resource_error(
            ResourceAdmissionError::UnavailableProvider,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unobserved_admission_preserves_applicability_and_closed_resource_causes() {
        let queue_limited = ProviderResourceSnapshot {
            queue_depth: Some(0),
            queue_limit: Some(2),
            ..Default::default()
        };
        assert!(require_queue_bound(&queue_limited, None).is_err());
        assert!(require_queue_bound(&queue_limited, Some(0)).is_ok());
        assert!(require_queue_bound(&ProviderResourceSnapshot::default(), None).is_ok());
        for host_budget in [None, Some(1024)] {
            for code in [
                ProviderRejectionCode::UnsupportedOperation,
                ProviderRejectionCode::ResourceLimit,
                ProviderRejectionCode::ProviderUnavailable,
                ProviderRejectionCode::UnknownResourceEstimate,
            ] {
                let result = estimate(
                    ProviderFeasibility::rejected(code, "fixture"),
                    &ProviderResourceSnapshot::default(),
                    host_budget,
                );
                if matches!(
                    code,
                    ProviderRejectionCode::UnsupportedOperation
                        | ProviderRejectionCode::UnknownResourceEstimate
                ) && host_budget.is_none()
                {
                    assert!(result.unwrap().is_none());
                } else {
                    let error = result.unwrap_err();
                    let cause = error
                        .source
                        .as_ref()
                        .unwrap()
                        .downcast_ref::<runmat_accelerate_api::ProviderRejection>()
                        .unwrap();
                    assert_eq!(cause.code, code);
                    assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
                }
            }
        }
    }
}
