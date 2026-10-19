use super::*;
use crate::builtins::common::test_support;
use runmat_accelerate_api::random::{
    GpuBoundedIntegerDomain, GpuIntegerRequest, GpuNormalTransform, GpuRandomAlgorithm,
    GpuRandomReservation,
};
use runmat_accelerate_api::{
    HostIntegerDataView, HostIntegerTensorView, IntegerElementType, NumericElementType,
};

fn upload_i32(
    provider: &'static dyn AccelProvider,
    values: &[i32],
    shape: &[usize],
) -> GpuTensorHandle {
    provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I32(values),
            shape,
        })
        .expect("upload exact i32 test allocation")
}

fn upload_u32(
    provider: &'static dyn AccelProvider,
    values: &[u32],
    shape: &[usize],
) -> GpuTensorHandle {
    provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::U32(values),
            shape,
        })
        .expect("upload exact u32 test allocation")
}

fn integer_request(shape: &[usize]) -> GpuIntegerRequest<'_> {
    GpuIntegerRequest {
        shape,
        class: NumericElementType::I32,
        domain: GpuBoundedIntegerDomain::new(-2, 7).expect("valid non-power-of-two domain"),
        reservation: GpuRandomReservation {
            algorithm: GpuRandomAlgorithm::Threefry4x64_20,
            seed: 19,
            first: 5,
            count: 4,
            normal_transform: GpuNormalTransform::BoxMuller,
        },
    }
}

fn pending(
    provider: &'static dyn AccelProvider,
    shape: &[usize],
    remaining: u64,
) -> GpuIntegerPending {
    GpuIntegerPending {
        output: upload_i32(provider, &[0; 4], shape),
        accepted: upload_u32(provider, &[0; 4], shape),
        remaining,
    }
}

#[test]
fn progress_count_is_bounded_without_requiring_strict_progress() {
    test_support::with_test_provider(|provider| {
        let shape = [1, 4];
        let source = upload_i32(provider, &[0; 4], &shape);
        let request = integer_request(&shape);

        let malformed = pending(provider, &shape, 5);
        let output = malformed.output.downgrade();
        let accepted = malformed.accepted.downgrade();
        let error = validate_progress(
            provider,
            &source,
            GpuIntegerBatchRequest {
                integer: request,
                first_attempt: 0,
                attempts_per_item: 8,
                pending: None,
            },
            GpuIntegerBatchProgress::Pending(malformed),
            16,
            16,
        )
        .err()
        .expect("a first batch cannot report more unresolved items than it owns");
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(output.acquire().is_err());
        assert!(accepted.acquire().is_err());

        let previous = pending(provider, &shape, 3);
        let output = previous.output.downgrade();
        let accepted = previous.accepted.downgrade();
        let increasing = GpuIntegerPending {
            output: previous.output.clone(),
            accepted: previous.accepted.clone(),
            remaining: 4,
        };
        let error = validate_progress(
            provider,
            &source,
            GpuIntegerBatchRequest {
                integer: request,
                first_attempt: 8,
                attempts_per_item: 8,
                pending: Some(&previous),
            },
            GpuIntegerBatchProgress::Pending(increasing),
            16,
            16,
        )
        .err()
        .expect("a continuation count cannot increase");
        assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
        assert!(output.acquire().is_ok(), "the prior owner remains live");
        assert!(accepted.acquire().is_ok(), "the prior mask remains live");
        drop(previous);
        assert!(output.acquire().is_err());
        assert!(accepted.acquire().is_err());

        let previous = pending(provider, &shape, 3);
        let output = previous.output.downgrade();
        let accepted = previous.accepted.downgrade();
        let unchanged = GpuIntegerPending {
            output: previous.output.clone(),
            accepted: previous.accepted.clone(),
            remaining: 3,
        };
        let (returned, complete) = validate_progress(
            provider,
            &source,
            GpuIntegerBatchRequest {
                integer: request,
                first_attempt: 16,
                attempts_per_item: 8,
                pending: Some(&previous),
            },
            GpuIntegerBatchProgress::Pending(unchanged),
            16,
            16,
        )
        .expect("a finite batch may accept no items");
        assert!(!complete);
        assert_eq!(returned.remaining, 3);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&returned.output),
            Some(IntegerElementType::I32)
        );
        drop(previous);
        assert!(
            output.acquire().is_ok(),
            "returned output keeps its owner live"
        );
        assert!(
            accepted.acquire().is_ok(),
            "returned mask keeps its owner live"
        );
        drop(returned);
        assert!(output.acquire().is_err());
        assert!(accepted.acquire().is_err());
    });
}
