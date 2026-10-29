//! Explicit adapter witnesses. Run separately with FORCE_PRECISION=f32/f64;
//! unavailable hardware/precision fails rather than silently skipping coverage.
use super::*;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    NumericElementType, ProviderPrecision, ProviderSpectralRange,
};

#[derive(Debug)]
struct EnvelopeScalarReadbackFault;

impl std::fmt::Display for EnvelopeScalarReadbackFault {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("envelope scalar readback fixture fault")
    }
}

impl std::error::Error for EnvelopeScalarReadbackFault {}

fn provider() -> WgpuProvider {
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter required");
    if let Ok(requested) = std::env::var("RUNMAT_WGPU_FORCE_PRECISION") {
        if requested == "f64" {
            assert_eq!(
                provider.precision(),
                ProviderPrecision::F64,
                "actual F64 device required"
            );
        } else if requested == "f32" {
            assert_eq!(provider.precision(), ProviderPrecision::F32);
        }
    }
    provider
}

fn upload(provider: &WgpuProvider, data: &[f64], shape: &[usize]) -> GpuTensorHandle {
    let single: Vec<_> = data.iter().map(|&value| value as f32).collect();
    let data = match provider.precision() {
        ProviderPrecision::F32 => HostNumericDataView::F32(&single),
        ProviderPrecision::F64 => HostNumericDataView::F64(data),
    };
    provider
        .upload_numeric(&HostNumericTensorView {
            data,
            shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

fn read(provider: &WgpuProvider, handle: &GpuTensorHandle) -> Vec<f64> {
    let result = pollster::block_on(provider.download_numeric(handle)).unwrap();
    assert_eq!(result.shape, handle.shape());
    assert_eq!(result.storage, GpuTensorStorage::Real);
    match (provider.precision(), result.data) {
        (ProviderPrecision::F32, HostNumericDataOwned::F32(values)) => {
            values.into_iter().map(f64::from).collect()
        }
        (ProviderPrecision::F64, HostNumericDataOwned::F64(values)) => values,
        _ => panic!("native output class changed"),
    }
}

fn envelope(
    provider: &WgpuProvider,
    input: &GpuTensorHandle,
    method: ProviderEnvelopeMethod,
) -> ProviderEnvelopeResult {
    pollster::block_on(provider.signal_envelope_exec(&ProviderEnvelopeRequest {
        input,
        channel_len: 3,
        channel_count: 2,
        output_shape: &[3, 2],
        method,
    }))
    .unwrap()
}

#[test]
#[ignore = "requires actual WGPU adapter; run explicitly for both native precisions"]
fn spectral_envelope_transpose_preserves_logical_order_and_retires_materialization() {
    let provider = provider();
    let baseline = provider.allocations.retained_bytes();
    let original = upload(&provider, &[1., 10., 2., 20., 3., 30.], &[2, 3]);
    let lazy = provider.transpose_exec(&original).unwrap();
    assert!(runmat_accelerate_api::handle_is_transposed(&lazy));
    let logical = upload(&provider, &[1., 2., 3., 10., 20., 30.], &[3, 2]);
    let input_bytes = provider.allocations.retained_bytes();
    for method in [
        ProviderEnvelopeMethod::Rms { window_len: 1 },
        ProviderEnvelopeMethod::AnalyticFir { filter_len: 3 },
    ] {
        let before = provider.telemetry_snapshot();
        let actual = envelope(&provider, &lazy, method);
        let after = provider.telemetry_snapshot();
        // The input-domain check reduces finiteness on device and reads one
        // native scalar. No signal payload may cross to the host to materialize
        // the transpose; account for the validation transfer separately.
        assert_eq!(
            after.download_bytes - before.download_bytes,
            provider.element_size as u64,
            "only the finite-validation scalar may download; materialization stays on device"
        );
        let expected_upload = match method {
            ProviderEnvelopeMethod::AnalyticFir { filter_len } => {
                filter_len as u64 * provider.element_size as u64
            }
            _ => 0,
        };
        assert_eq!(
            after.upload_bytes - before.upload_bytes,
            expected_upload,
            "only FIR coefficients may upload"
        );
        for output in [&actual.upper, &actual.lower] {
            assert_eq!(output.shape(), &[3, 2]);
            assert!(!runmat_accelerate_api::handle_is_transposed(output));
            assert!(!output.shares_allocation_with(&lazy));
            assert_eq!(
                output.physical().element_type(),
                match provider.precision() {
                    ProviderPrecision::F32 => NumericElementType::F32,
                    ProviderPrecision::F64 => NumericElementType::F64,
                }
            );
        }
        assert!(!actual.upper.shares_allocation_with(&actual.lower));
        let expected = envelope(&provider, &logical, method);
        let actual_upper = read(&provider, &actual.upper);
        let actual_lower = read(&provider, &actual.lower);
        for (actual, expected) in actual_upper.iter().zip(read(&provider, &expected.upper)) {
            assert!((actual - expected).abs() <= 1e-5 * expected.abs().max(1.));
        }
        for (actual, expected) in actual_lower.iter().zip(read(&provider, &expected.lower)) {
            assert!((actual - expected).abs() <= 1e-5 * expected.abs().max(1.));
        }
        if matches!(method, ProviderEnvelopeMethod::Rms { .. }) {
            assert_eq!(actual_upper, vec![1., 2., 3., 10., 20., 30.]);
        }
        drop((actual, expected));
        assert_eq!(
            provider.allocations.retained_bytes(),
            input_bytes,
            "no kernel/materialization lease leaked"
        );
        assert!(runmat_accelerate_api::handle_is_transposed(&lazy));
        original.allocation_lease().validate_live().unwrap();
    }
    drop((logical, lazy, original));
    assert_eq!(provider.allocations.retained_bytes(), baseline);
}

fn request(input: &GpuTensorHandle) -> ProviderSpectralRequest<'_> {
    ProviderSpectralRequest {
        input,
        input_len: 16,
        input_complex: false,
        window: &[1.; 4],
        nfft: 8,
        frame_count: 4,
        frame_mode: ProviderSpectralFrameMode::ColumnSliding {
            hop: 4,
            input_rows: 8,
            frames_per_column: 2,
        },
        range: ProviderSpectralRange::Onesided,
        denominator: 4.,
    }
}

#[test]
#[ignore = "requires actual WGPU adapter; no unavailable-device skip"]
fn spectral_invalid_request_and_foreign_owner_have_zero_upload_or_dispatch() {
    use runmat_accelerate_api::observation::{
        with_provider_observation, ProviderObservationStatus, ProviderOperationReceipt,
    };
    let provider = provider();
    let input = upload(&provider, &[1.; 16], &[8, 2]);
    let foreign_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let foreign = foreign_owner
        .allocate_zeroed(
            vec![8, 2],
            input.physical().element_type(),
            GpuTensorStorage::Real,
        )
        .unwrap();
    let mut malformed = request(&input);
    malformed.frame_count = 0;
    let mut overflow = request(&input);
    overflow.nfft = usize::MAX;
    let mut cross_column = request(&input);
    cross_column.frame_count = 2;
    cross_column.frame_mode = ProviderSpectralFrameMode::ColumnSliding {
        hop: 6,
        input_rows: 8,
        frames_per_column: 2,
    };
    let mut folded_cross_column = request(&input);
    folded_cross_column.frame_count = 2;
    folded_cross_column.frame_mode = ProviderSpectralFrameMode::FoldedColumns { input_rows: 2 };
    for request in [
        malformed,
        overflow,
        cross_column,
        folded_cross_column,
        request(&foreign.handle),
    ] {
        let before = provider.telemetry_snapshot();
        let bytes = provider.allocations.retained_bytes();
        let mut receipt = ProviderOperationReceipt::new();
        let result = with_provider_observation(Some(receipt.recorder()), || {
            pollster::block_on(provider.uniform_spectral_estimate_exec(&request))
        });
        assert!(result.is_err());
        receipt.publish().unwrap();
        assert!(matches!(
            receipt.take_completed().unwrap().status,
            ProviderObservationStatus::Completed { submissions: 0, .. }
        ));
        assert_eq!(
            provider.telemetry_snapshot().upload_bytes,
            before.upload_bytes
        );
        assert_eq!(provider.allocations.retained_bytes(), bytes);
        input.allocation_lease().validate_live().unwrap();
    }
    assert_eq!(foreign.probe.retirements(), 0);
}

#[test]
#[ignore = "requires actual WGPU adapter; no unavailable-device skip"]
fn envelope_materialization_failure_and_unpolled_drop_retain_original_only() {
    let mut provider = provider();
    let original = upload(&provider, &[1., 10., 2., 20., 3., 30.], &[2, 3]);
    let lazy = provider.transpose_exec(&original).unwrap();
    let bytes = provider.allocations.retained_bytes();
    let before = provider.telemetry_snapshot();
    let request = ProviderEnvelopeRequest {
        input: &lazy,
        channel_len: 3,
        channel_count: 2,
        output_shape: &[3, 2],
        method: ProviderEnvelopeMethod::Rms { window_len: 1 },
    };
    drop(provider.signal_envelope_exec(&request));
    assert_eq!(
        provider.telemetry_snapshot().upload_bytes,
        before.upload_bytes
    );
    assert_eq!(provider.allocations.retained_bytes(), bytes);
    // Permutation uses two storage bindings; bounds needs three. This forces
    // failure after preparation without changing real device resource authority.
    provider.adapter_limits.max_storage_buffers_per_shader_stage = 2;
    assert!(pollster::block_on(provider.signal_envelope_exec(&request)).is_err());
    assert_eq!(provider.allocations.retained_bytes(), bytes);
    assert!(runmat_accelerate_api::handle_is_transposed(&lazy));
    assert_eq!(read(&provider, &original), vec![1., 10., 2., 20., 3., 30.]);
    let next = upload(&provider, &[9.], &[1, 1]);
    assert!(
        next.buffer_id() > original.buffer_id() + 1,
        "failure happened after a distinct materialization was registered"
    );
}

#[test]
#[ignore = "requires actual WGPU adapter; envelope validation owns real mask/reduction buffers"]
fn envelope_scalar_readback_failure_and_pending_drop_retire_validation_intermediates() {
    use std::future::Future;
    use std::task::{Context, Poll};

    let provider = provider();
    let input = upload(&provider, &[1., 2., 3., 4., 5., 6.], &[3, 2]);
    let retained_input = provider.allocations.retained_bytes();

    let failed = provider
        .prepare_envelope_finite_validation(&input)
        .expect("real finite mask and scalar reduction");
    let scalar_id = failed.scalar().buffer_id();
    assert!(provider.allocations.retained_bytes() > retained_input);
    let error = failed
        .complete(Err(anyhow::Error::new(EnvelopeScalarReadbackFault)))
        .expect_err("scalar readback fault is terminal after device effects");
    assert!(error.is_after_effect());
    assert!(std::error::Error::source(&error)
        .and_then(|source| source.downcast_ref::<EnvelopeScalarReadbackFault>())
        .is_some());
    assert_eq!(provider.allocations.retained_bytes(), retained_input);
    input.allocation_lease().validate_live().unwrap();
    let next = upload(&provider, &[9.], &[1, 1]);
    assert_eq!(
        next.buffer_id(),
        scalar_id + 1,
        "failed validation must not register either envelope bounds output"
    );
    drop(next);
    assert_eq!(provider.allocations.retained_bytes(), retained_input);

    let pending = provider
        .prepare_envelope_finite_validation(&input)
        .expect("second real finite mask and scalar reduction");
    let mut future = Box::pin(async move {
        futures::future::pending::<()>().await;
        pending.scalar().buffer_id()
    });
    let waker = futures::task::noop_waker();
    assert!(matches!(
        future.as_mut().poll(&mut Context::from_waker(&waker)),
        Poll::Pending
    ));
    assert!(provider.allocations.retained_bytes() > retained_input);
    drop(future);
    assert_eq!(provider.allocations.retained_bytes(), retained_input);
    input.allocation_lease().validate_live().unwrap();
    assert_eq!(read(&provider, &input), vec![1., 2., 3., 4., 5., 6.]);
}
