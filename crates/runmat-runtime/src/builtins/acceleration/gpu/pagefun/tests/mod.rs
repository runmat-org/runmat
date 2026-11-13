use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::HostTensorView;
use runmat_value::{CharArray, IntegerStorage, StringArray};

#[test]
fn product_operational_failures_keep_pagefun_identity_without_losing_retry() {
    use crate::builtins::common::gpu_helpers::ResidentComputeDecline;
    for (decline, retry) in [
        (
            ResidentComputeDecline::PolicyDenied,
            crate::GpuGatherRetry::Requested,
        ),
        (
            ResidentComputeDecline::InvalidOwner,
            crate::GpuGatherRetry::Never,
        ),
    ] {
        let original = crate::build_runtime_error("shared product failure")
            .with_source(std::io::Error::other("retained provider cause"))
            .build();
        let mapped = pagefun_operational_error(decline.with_domain_error(original));
        assert_eq!(mapped.identifier(), PAGEFUN_ERROR_INTERNAL.identifier);
        assert_eq!(mapped.context.builtin.as_deref(), Some("pagefun"));
        assert_eq!(mapped.gpu_gather_retry, retry);
        let mut cause = std::error::Error::source(&mapped);
        let mut retained = false;
        while let Some(error) = cause {
            retained |= error.downcast_ref::<std::io::Error>().is_some();
            cause = error.source();
        }
        assert!(retained);
    }
    let original = crate::build_runtime_error("cancelled")
        .with_identifier("RunMat:Cancelled")
        .with_failure_origin(
            crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation,
        )
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build();
    let mapped = pagefun_operational_error(original);
    assert_eq!(mapped.identifier(), Some("RunMat:Cancelled"));
    assert_eq!(
        mapped.failure_origin(),
        crate::runtime_error::RuntimeFailureOrigin::ExecutionControlCancellation
    );
    assert_eq!(mapped.gpu_gather_retry, crate::GpuGatherRetry::Never);
    let specific = crate::build_runtime_error("provider integrity failure")
        .with_identifier("RunMat:Provider:Integrity")
        .build();
    assert_eq!(
        pagefun_operational_error(specific).identifier(),
        Some("RunMat:Provider:Integrity")
    );
}

mod host;
mod integer;
mod provider;
mod selectors;
mod provider_contract;
mod cutover;
