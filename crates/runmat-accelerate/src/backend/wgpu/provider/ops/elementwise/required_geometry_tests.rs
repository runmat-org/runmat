//! Actual-adapter regression for the ordinary fused physical-word guard.
use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataView, HostNumericTensorView, NumericElementType,
    ProviderPrecision,
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
    // SAFETY: this independently created device owner lives for the test process/worker.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn upload(provider: &WgpuProvider, values: &[f32], shape: &[usize]) -> GpuTensorHandle {
    provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(values),
            shape,
            storage: GpuTensorStorage::Real,
        })
        .expect("real input upload")
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
    let words = shape.iter().product::<usize>() * lanes;
    assert_eq!(expected.len(), words);
    assert_eq!(
        handle.physical().requested_bytes(),
        (words.max(1) * 4) as u64
    );
    let host = provider
        .download(handle)
        .await
        .expect("actual GPU readback");
    assert_eq!(host.shape, shape);
    assert_eq!(host.storage, storage);
    assert_eq!(host.data.len(), words);
    for (&actual, &expected) in host.data.iter().zip(expected) {
        assert!((actual - expected).abs() <= 2e-4, "{actual} != {expected}");
    }
}

async fn exercise() {
    let provider = owner().await;
    let real = upload(provider, &[1.0, 2.0], &[2, 1]);
    let imag = upload(provider, &[3.0, 4.0], &[2, 1]);
    let from_real = provider.complex_from_real(&real).await.unwrap();
    check(
        provider,
        &from_real,
        &[2, 1],
        GpuTensorStorage::ComplexInterleaved,
        &[1., 0., 2., 0.],
    )
    .await;
    let complex = provider.complex_from_real_imag(&real, &imag).await.unwrap();
    check(
        provider,
        &complex,
        &[2, 1],
        GpuTensorStorage::ComplexInterleaved,
        &[1., 3., 2., 4.],
    )
    .await;
    let conjugate = provider
        .unary_conj(&complex)
        .await
        .unwrap()
        .expect("admitted conjugation");
    check(
        provider,
        &conjugate,
        &[2, 1],
        GpuTensorStorage::ComplexInterleaved,
        &[1., -3., 2., -4.],
    )
    .await;
    let real_part = provider.unary_real(&complex).await.unwrap();
    check(
        provider,
        &real_part,
        &[2, 1],
        GpuTensorStorage::Real,
        &[1., 2.],
    )
    .await;
    let imag_part = provider.unary_imag(&complex).await.unwrap();
    check(
        provider,
        &imag_part,
        &[2, 1],
        GpuTensorStorage::Real,
        &[3., 4.],
    )
    .await;
    let magnitude = provider.unary_abs(&complex).await.unwrap();
    check(
        provider,
        &magnitude,
        &[2, 1],
        GpuTensorStorage::Real,
        &[10f64.sqrt(), 20f64.sqrt()],
    )
    .await;
    let sum = provider.elem_add(&complex, &complex).await.unwrap();
    check(
        provider,
        &sum,
        &[2, 1],
        GpuTensorStorage::ComplexInterleaved,
        &[2., 6., 4., 8.],
    )
    .await;
    let scalar_re = upload(provider, &[5.0], &[1, 1]);
    let scalar_im = upload(provider, &[-1.0], &[1, 1]);
    let scalar = provider
        .complex_from_real_imag(&scalar_re, &scalar_im)
        .await
        .unwrap();
    let broadcast = provider.elem_add(&complex, &scalar).await.unwrap();
    check(
        provider,
        &broadcast,
        &[2, 1],
        GpuTensorStorage::ComplexInterleaved,
        &[6., 2., 7., 3.],
    )
    .await;
    let empty_re = upload(provider, &[], &[0, 1]);
    let empty_im = upload(provider, &[], &[0, 1]);
    let empty = provider
        .complex_from_real_imag(&empty_re, &empty_im)
        .await
        .unwrap();
    let empty_sum = provider.elem_add(&empty, &empty).await.unwrap();
    check(
        provider,
        &empty_sum,
        &[0, 1],
        GpuTensorStorage::ComplexInterleaved,
        &[],
    )
    .await;
    for borrowed in [&real, &imag, &complex, &scalar, &empty] {
        borrowed.allocation_lease().validate_live().unwrap();
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual F32 WGPU adapter"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn required_real_complex_unary_binary_broadcast_and_empty_geometry() {
    exercise().await;
}
