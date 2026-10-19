use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    NumericElementType, ProviderPrecision,
};

#[test]
#[ignore = "requires a real shader-f64 WGPU adapter; F32 physical execution on its selected F64 owner"]
fn actual_f64_provider_executes_f32_real_binary_scalar_unary_without_reinterpretation() {
    let mut provider = crate::backend::wgpu::provider::WgpuProvider::new(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual shader-f64 WGPU adapter is required");
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.0, 4.0, 9.0, 16.0]),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let weak = input.downgrade();
    assert!(
        provider.get_entry(&input).is_err(),
        "native F64 entry gate remains strict"
    );
    let scalar = provider.scalar_add(&input, 1.0).unwrap();
    let unary = block_on(provider.unary_sqrt(&input)).unwrap();
    let same = block_on(provider.elem_add(&input, &unary)).unwrap();
    let control = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[10.0, 20.0]),
            shape: &[2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let broadcast = block_on(provider.elem_add(&input, &control)).unwrap();
    let native = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[1.0, 4.0, 9.0, 16.0]),
            shape: &[2, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(block_on(provider.elem_add(&input, &native)).is_err());
    for (handle, expected) in [
        (&scalar, &[2.0_f32, 5.0, 10.0, 17.0][..]),
        (&unary, &[1.0_f32, 2.0, 3.0, 4.0][..]),
        (&same, &[2.0_f32, 6.0, 12.0, 20.0][..]),
        (&broadcast, &[11.0_f32, 24.0, 19.0, 36.0][..]),
    ] {
        assert_eq!(handle.shape(), [2, 2]);
        assert_eq!(handle.physical().element_type(), NumericElementType::F32);
        assert!(!handle.shares_allocation_with(&input));
        handle.allocation_lease().validate_live().unwrap();
        assert_eq!(
            block_on(provider.download_numeric(handle)).unwrap().data,
            HostNumericDataOwned::F32(expected.to_vec())
        );
    }
    let before_rejected = provider.telemetry_snapshot();
    provider.adapter_limits.max_buffer_size = 8;
    let oversized = block_on(provider.elem_add(&input, &unary)).unwrap_err();
    assert!(oversized.is_unsupported());
    let after_rejected = provider.telemetry_snapshot();
    assert_eq!(
        after_rejected.fused_elementwise.count, before_rejected.fused_elementwise.count,
        "oversized selected F32 output must reject before any binary dispatch"
    );
    assert_eq!(
        after_rejected.download_bytes,
        before_rejected.download_bytes
    );
    input.allocation_lease().validate_live().unwrap();
    drop(broadcast);
    drop(control);
    drop(same);
    drop(unary);
    drop(scalar);
    drop(native);
    drop(input);
    assert!(weak.acquire().is_err());
}
