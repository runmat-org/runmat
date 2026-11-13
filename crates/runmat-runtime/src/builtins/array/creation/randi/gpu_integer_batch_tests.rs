use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_accelerate_api::random::{
    GpuBoundedIntegerDomain, GpuIntegerBatchProgress, GpuIntegerBatchRequest, GpuIntegerRequest,
    GpuNormalTransform, GpuRandomAlgorithm, GpuRandomReservation,
};
use runmat_accelerate_api::{AccelProvider as _, HostIntegerDataOwned, NumericElementType};

fn expected_values(request: GpuIntegerRequest<'_>, attempts: u32) -> Vec<i128> {
    (0..request.reservation.count)
        .map(|item| {
            (0..u64::from(attempts))
                .find_map(|attempt| {
                    request
                        .domain
                        .sample_attempt(request.reservation, item, attempt)
                        .expect("valid independent scalar attempt")
                })
                .expect("selected deterministic witness completes in one finite batch")
        })
        .collect()
}

#[test]
#[ignore = "run explicitly on actual WGPU hardware"]
fn actual_wgpu_non_power_of_two_integer_batches_match_scalar_child_stream() {
    let _guard = test_support::accel_test_lock();
    let provider =
        runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default())
            .expect("actual WGPU adapter required");
    let _thread = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let shape = [3, 43];
    let attempts = 8;

    for (class, lower, span, seed) in [
        (NumericElementType::I32, -50, 101, 41),
        (NumericElementType::U32, 4_000_000_000, 251, 43),
    ] {
        let request = GpuIntegerRequest {
            shape: &shape,
            class,
            domain: GpuBoundedIntegerDomain::new(lower, span)
                .expect("valid non-power-of-two integer domain"),
            reservation: GpuRandomReservation {
                algorithm: GpuRandomAlgorithm::Threefry4x64_20,
                seed,
                first: 29,
                count: 129,
                normal_transform: GpuNormalTransform::BoxMuller,
            },
        };
        assert!(request.fixed_work_32().is_none());
        assert!(request.rejection_domain_32().is_some());
        let expected = expected_values(request, attempts);
        let batch = GpuIntegerBatchRequest {
            integer: request,
            first_attempt: 0,
            attempts_per_item: attempts,
            pending: None,
        };
        provider
            .propose_gpu_integer_batch(batch)
            .expect("physical proposal")
            .expect("actual WGPU provider must support the selected batch");
        let progress = block_on(provider.execute_gpu_integer_batch(batch, None, None))
            .expect("execute actual WGPU integer batch");
        let GpuIntegerBatchProgress::Complete(completed) = progress else {
            panic!("deterministic witness must complete its acknowledged batch");
        };
        assert_eq!(completed.remaining, 0);
        assert_eq!(completed.output.shape(), shape);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&completed.output),
            class.integer_type()
        );

        let downloaded = block_on(provider.download_integer(&completed.output))
            .expect("download exact integer output");
        assert_eq!(downloaded.shape, shape);
        let observed: Vec<i128> = match downloaded.data {
            HostIntegerDataOwned::I32(values) => values.into_iter().map(i128::from).collect(),
            HostIntegerDataOwned::U32(values) => values.into_iter().map(i128::from).collect(),
            other => panic!("unexpected actual WGPU integer class: {other:?}"),
        };
        assert_eq!(observed, expected);
        completed
            .accepted
            .release()
            .expect("retire unpublished acceptance mask");
        completed
            .output
            .release()
            .expect("retire downloaded test output");
    }
}
