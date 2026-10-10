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
    let bounds = provider
        .signal_envelope(&ProviderEnvelopeRequest {
            input: &constant,
            channel_len: 8,
            channel_count: 1,
            output_shape: &[8, 1],
            method: ProviderEnvelopeMethod::Analytic,
        })
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
