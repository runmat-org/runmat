//! Required actual-adapter spectral and analytic-signal geometry witnesses.
use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataView, HostNumericTensorView, NumericElementType,
    ProviderPrecision, ProviderSpectralFrameMode, ProviderSpectralRange,
};

async fn owner() -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let provider = WgpuProvider::new(WgpuProviderOptions::default()).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(WgpuProviderOptions::default())
        .await
        .expect("actual WebGPU adapter");
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let provider = Box::leak(Box::new(provider));
    // SAFETY: this actual device owner lives for the test process/worker.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn upload(provider: &WgpuProvider, data: &[f32], shape: &[usize]) -> GpuTensorHandle {
    provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(data),
            shape,
            storage: GpuTensorStorage::Real,
        })
        .expect("real source upload")
}

async fn check(
    provider: &WgpuProvider,
    handle: &GpuTensorHandle,
    shape: &[usize],
    storage: GpuTensorStorage,
    expected: &[f64],
) {
    assert_eq!(handle.shape(), shape);
    assert_eq!(handle.physical().storage(), storage);
    assert_eq!(handle.physical().element_type(), NumericElementType::F32);
    let lanes = if storage == GpuTensorStorage::ComplexInterleaved {
        2
    } else {
        1
    };
    assert_eq!(expected.len(), shape.iter().product::<usize>() * lanes);
    assert_eq!(
        handle.physical().requested_bytes(),
        (expected.len().max(1) * 4) as u64
    );
    let host = provider
        .download(handle)
        .await
        .expect("actual GPU readback");
    assert_eq!(host.shape, shape);
    assert_eq!(host.storage, storage);
    assert_eq!(host.data.len(), expected.len());
    for (&actual, &expected) in host.data.iter().zip(expected) {
        assert!((actual - expected).abs() <= 2e-4, "{actual} != {expected}");
    }
}

async fn exercise() {
    let provider = owner().await;
    let impulse = upload(provider, &[1., 0., 0., 0.], &[4, 1]);
    let estimate = provider
        .uniform_spectral_estimate(&ProviderSpectralRequest {
            input: &impulse,
            input_len: 4,
            input_complex: false,
            window: &[1.; 4],
            nfft: 4,
            frame_count: 1,
            frame_mode: ProviderSpectralFrameMode::Sliding { hop: 4 },
            range: ProviderSpectralRange::Onesided,
            denominator: 4.,
        })
        .await
        .expect("frame, FFT, select, and power on GPU");
    assert_eq!((estimate.rows, estimate.cols), (3, 1));
    check(
        provider,
        &estimate.s,
        &[3, 1],
        GpuTensorStorage::ComplexInterleaved,
        &[1., 0., 1., 0., 1., 0.],
    )
    .await;
    check(
        provider,
        &estimate.ps,
        &[3, 1],
        GpuTensorStorage::Real,
        &[0.25, 0.5, 0.25],
    )
    .await;
    assert!(!estimate.s.shares_allocation_with(&impulse));
    assert!(!estimate.ps.shares_allocation_with(&impulse));
    impulse.allocation_lease().validate_live().unwrap();

    let wave = upload(provider, &[1., 0., -1., 0.], &[1, 4]);
    let analytic = provider
        .signal_hilbert(&ProviderHilbertRequest {
            input: &wave,
            length: None,
            dim: 1,
        })
        .await
        .expect("Hilbert analytic signal on GPU");
    check(
        provider,
        &analytic,
        &[1, 4],
        GpuTensorStorage::ComplexInterleaved,
        &[1., 0., 0., 1., -1., 0., 0., -1.],
    )
    .await;
    wave.allocation_lease().validate_live().unwrap();

    let constant = upload(provider, &[2.5; 8], &[8, 1]);
    let supported = ProviderEnvelopeRequest {
        input: &constant,
        channel_len: 8,
        channel_count: 1,
        output_shape: &[8, 1],
        method: ProviderEnvelopeMethod::Analytic,
    };
    assert!(provider.supports_signal_envelope(&supported));
    assert!(
        !provider.supports_signal_envelope(&ProviderEnvelopeRequest {
            input: &constant,
            channel_len: 8,
            channel_count: 1,
            output_shape: &[8, 1],
            method: ProviderEnvelopeMethod::Rms { window_len: 0 },
        })
    );
    let bounds = provider
        .signal_envelope(&supported)
        .await
        .expect("analytic envelope on GPU");
    check(
        provider,
        &bounds.upper,
        &[8, 1],
        GpuTensorStorage::Real,
        &[2.5; 8],
    )
    .await;
    check(
        provider,
        &bounds.lower,
        &[8, 1],
        GpuTensorStorage::Real,
        &[2.5; 8],
    )
    .await;
    assert!(!bounds.upper.shares_allocation_with(&constant));
    assert!(!bounds.lower.shares_allocation_with(&constant));
    constant.allocation_lease().validate_live().unwrap();

    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let source = upload(provider, &[1.0, invalid, 3.0, 4.0], &[4, 1]);
        let request = ProviderEnvelopeRequest {
            input: &source,
            channel_len: 4,
            channel_count: 1,
            output_shape: &[4, 1],
            method: ProviderEnvelopeMethod::Rms { window_len: 2 },
        };
        assert!(provider.supports_signal_envelope(&request));
        let retained = provider.allocations.retained_bytes();
        let error = provider
            .signal_envelope(&request)
            .await
            .expect_err("resident nonfinite signal must reject before bounds");
        assert!(error.is_after_effect());
        assert!(!error.is_unsupported());
        assert!(matches!(
            std::error::Error::source(&error)
                .and_then(|source| source
                    .downcast_ref::<runmat_accelerate_api::EnvelopeSignalDomainError>()),
            Some(runmat_accelerate_api::EnvelopeSignalDomainError::NonFinite)
        ));
        assert_eq!(provider.allocations.retained_bytes(), retained);
        source.allocation_lease().validate_live().unwrap();
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires actual F32 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f32; a foreign allocation must preserve its owner error"]
fn direct_hilbert_foreign_owner_preserves_acquisition_cause() {
    use runmat_accelerate_api::allocation::AllocationAcquisitionError;
    use runmat_accelerate_api::test_support::TestAllocationOwner;

    let provider = pollster::block_on(owner());
    let foreign = TestAllocationOwner::new()
        .allocate_zeroed(vec![1, 4], NumericElementType::F32, GpuTensorStorage::Real)
        .expect("foreign live allocation");
    let request = ProviderHilbertRequest {
        input: &foreign.handle,
        length: None,
        dim: 1,
    };
    assert!(!provider.supports_signal_hilbert(&request));
    let error = pollster::block_on(provider.signal_hilbert(&request))
        .expect_err("direct hook must authenticate its owner before shape admission");
    assert!(!error.is_unsupported());
    assert!(!error.is_after_effect());
    assert!(matches!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<AllocationAcquisitionError>()),
        Some(AllocationAcquisitionError::WrongOwner)
    ));
    assert_eq!(foreign.probe.retirements(), 0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires actual F32 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f32; foreign envelope input must retain owner cause"]
fn direct_envelope_foreign_owner_preserves_acquisition_cause() {
    use runmat_accelerate_api::allocation::AllocationAcquisitionError;
    use runmat_accelerate_api::test_support::TestAllocationOwner;

    let provider = pollster::block_on(owner());
    let foreign = TestAllocationOwner::new()
        .allocate_zeroed(vec![4, 1], NumericElementType::F32, GpuTensorStorage::Real)
        .expect("foreign live input");
    let request = ProviderEnvelopeRequest {
        input: &foreign.handle,
        channel_len: 4,
        channel_count: 1,
        output_shape: &[4, 1],
        method: ProviderEnvelopeMethod::Rms { window_len: 2 },
    };
    assert!(!provider.supports_signal_envelope(&request));
    let error = pollster::block_on(provider.signal_envelope(&request))
        .expect_err("direct hook must authenticate the allocation");
    assert!(!error.is_unsupported());
    assert!(!error.is_after_effect());
    assert!(matches!(
        std::error::Error::source(&error)
            .and_then(|source| source.downcast_ref::<AllocationAcquisitionError>()),
        Some(AllocationAcquisitionError::WrongOwner)
    ));
    assert_eq!(foreign.probe.retirements(), 0);
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual F32 WGPU adapter"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn required_spectral_frame_select_power_hilbert_and_envelope_geometry() {
    exercise().await;
}
