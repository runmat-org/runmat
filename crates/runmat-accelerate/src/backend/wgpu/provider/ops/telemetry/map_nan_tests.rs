use super::*;
use runmat_accelerate_api::{
    GpuTensorHandle, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    NumericElementType, WgpuBufferOwner,
};
use std::error::Error as _;

#[cfg(not(target_arch = "wasm32"))]
type WeakBuffer = std::sync::Weak<wgpu::Buffer>;
#[cfg(target_arch = "wasm32")]
type WeakBuffer = std::rc::Weak<wgpu::Buffer>;

fn weak_buffer(buffer: &WgpuBufferOwner) -> WeakBuffer {
    #[cfg(not(target_arch = "wasm32"))]
    {
        runmat_accelerate_api::WgpuBufferOwner::downgrade(buffer)
    }
    #[cfg(target_arch = "wasm32")]
    {
        runmat_accelerate_api::WgpuBufferOwner::downgrade(buffer)
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual F32 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f32"]
fn nan_map_postallocation_unsupported_is_terminal_and_drops_unpublished_buffer() {
    let provider = crate::backend::wgpu::provider::register_wgpu_provider(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU adapter required for NaN-map fault qualification");
    futures::executor::block_on(verify_nan_map_and_postallocation_failure(provider));
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_nan_map_drops_unpublished_output_after_typed_fault() {
    let provider =
        WgpuProvider::new_async(crate::backend::wgpu::provider::WgpuProviderOptions::default())
            .await
            .expect("this browser qualification requires a real WebGPU adapter/device");
    verify_nan_map_and_postallocation_failure(&provider).await;
}

async fn verify_nan_map_and_postallocation_failure(provider: &WgpuProvider) {
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    // The first two elements retain the original [NaN, 2] regression. The
    // remaining bit patterns distinguish NaN payload/sign from infinity,
    // signed zero, and subnormal inputs without a float-comparison oracle.
    let input_bits = [
        0x7fc0_0000,
        0x4000_0000,
        0x7fc0_1234,
        0xffc0_5678,
        0x7f80_0001,
        0xff80_0001,
        0x7f80_0000,
        0xff80_0000,
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
    ];
    let expected_map_bits = [
        0,
        0x4000_0000,
        0,
        0,
        0,
        0,
        0x7f80_0000,
        0xff80_0000,
        0,
        0x8000_0000,
        0x0000_0001,
        0x8000_0001,
    ];
    let expected_mask_bits = [
        0,
        0x3f80_0000,
        0,
        0,
        0,
        0,
        0x3f80_0000,
        0x3f80_0000,
        0x3f80_0000,
        0x3f80_0000,
        0x3f80_0000,
        0x3f80_0000,
    ];
    let values: Vec<f32> = input_bits.into_iter().map(f32::from_bits).collect();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[1, 12],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let cleaned = provider
        .map_nan_to_zero(&input)
        .expect("actual F32 NaN map executes on the selected owner");
    assert_eq!(cleaned.shape(), &[1, 12]);
    assert_eq!(
        cleaned.descriptor().element_type,
        Some(NumericElementType::F32)
    );
    assert_eq!(cleaned.descriptor().storage, Some(GpuTensorStorage::Real));
    assert!(!cleaned.shares_allocation_with(&input));
    let HostNumericDataOwned::F32(mapped) = provider.download_numeric(&cleaned).await.unwrap().data
    else {
        panic!("F32 NaN map must preserve its numeric class");
    };
    assert_eq!(
        mapped
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
            .as_slice(),
        expected_map_bits.as_slice()
    );
    let mask = provider
        .not_nan_mask(&input)
        .expect("the same IEEE predicate drives the numeric not-NaN mask");
    assert_eq!(mask.shape(), &[1, 12]);
    assert_eq!(
        mask.descriptor().element_type,
        Some(NumericElementType::F32)
    );
    assert_eq!(mask.descriptor().storage, Some(GpuTensorStorage::Real));
    assert!(!mask.shares_allocation_with(&input));
    assert!(!mask.shares_allocation_with(&cleaned));
    let HostNumericDataOwned::F32(mask_values) =
        provider.download_numeric(&mask).await.unwrap().data
    else {
        panic!("F32 not-NaN mask must retain its numeric class");
    };
    assert_eq!(
        mask_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
            .as_slice(),
        expected_mask_bits.as_slice()
    );
    mask.release().unwrap();
    cleaned.release().unwrap();
    input.allocation_lease().validate_live().unwrap();
    assert_postallocation_fault(provider, &input);
    input.release().unwrap();
}

fn assert_postallocation_fault(provider: &WgpuProvider, input: &GpuTensorHandle) {
    let records_before = provider.allocations.record_count().unwrap();
    let mut produced_buffer = None;
    let error = provider
        .map_nan_to_zero_exec_typed_after_allocation(input, |output| {
            produced_buffer = Some(weak_buffer(output));
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "NaN-map fault after output allocation",
            ))
        })
        .expect_err("an output allocation has crossed the no-replay boundary");
    assert!(!error.is_unsupported());
    assert!(error.source().is_some_and(|source| {
        source.is::<runmat_accelerate_api::UnsupportedProviderOperation>()
    }));
    assert!(produced_buffer
        .expect("the real allocation reached the checkpoint")
        .upgrade()
        .is_none());
    assert_eq!(provider.allocations.record_count().unwrap(), records_before);
    input.allocation_lease().validate_live().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter and RUNMAT_WGPU_FORCE_PRECISION=f64"]
fn f64_nan_map_and_mask_classify_low_mantissa_payloads_without_losing_finite_bits() {
    let provider =
        WgpuProvider::new(crate::backend::wgpu::provider::WgpuProviderOptions::default())
            .expect("this gate requires an actual shader-f64 WGPU adapter/device");
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    futures::executor::block_on(verify_f64_nan_map_and_mask(&provider));
}

#[cfg(not(target_arch = "wasm32"))]
async fn verify_f64_nan_map_and_mask(provider: &WgpuProvider) {
    // The third and fourth cases have a zero high mantissa word and only a
    // nonzero low word; a high-word-only NaN test would miss both.
    let input_bits = [
        0x7ff8_0000_0000_0000,
        0x4000_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0xfff0_0000_0000_0001,
        0x7ff8_0000_0000_1234,
        0xfff8_0000_0000_5678,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x8000_0000_0000_0001,
    ];
    let expected_map_bits = [
        0,
        0x4000_0000_0000_0000,
        0,
        0,
        0,
        0,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x8000_0000_0000_0001,
    ];
    let expected_mask_bits = [
        0,
        0x3ff0_0000_0000_0000,
        0,
        0,
        0,
        0,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x3ff0_0000_0000_0000,
    ];
    let values: Vec<f64> = input_bits.into_iter().map(f64::from_bits).collect();
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&values),
            shape: &[1, 12],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let cleaned = provider.map_nan_to_zero(&input).unwrap();
    assert_eq!(cleaned.shape(), &[1, 12]);
    assert_eq!(
        cleaned.descriptor().element_type,
        Some(NumericElementType::F64)
    );
    assert_eq!(cleaned.descriptor().storage, Some(GpuTensorStorage::Real));
    assert!(!cleaned.shares_allocation_with(&input));
    let HostNumericDataOwned::F64(mapped) = provider.download_numeric(&cleaned).await.unwrap().data
    else {
        panic!("F64 NaN map must preserve its numeric class");
    };
    assert_eq!(
        mapped
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
            .as_slice(),
        expected_map_bits.as_slice()
    );
    let mask = provider.not_nan_mask(&input).unwrap();
    assert_eq!(mask.shape(), &[1, 12]);
    assert_eq!(
        mask.descriptor().element_type,
        Some(NumericElementType::F64)
    );
    assert_eq!(mask.descriptor().storage, Some(GpuTensorStorage::Real));
    assert!(!mask.shares_allocation_with(&input));
    assert!(!mask.shares_allocation_with(&cleaned));
    let HostNumericDataOwned::F64(mask_values) =
        provider.download_numeric(&mask).await.unwrap().data
    else {
        panic!("F64 not-NaN mask must retain its numeric class");
    };
    assert_eq!(
        mask_values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
            .as_slice(),
        expected_mask_bits.as_slice()
    );
    mask.release().unwrap();
    cleaned.release().unwrap();
    input.allocation_lease().validate_live().unwrap();
    assert_postallocation_fault(provider, &input);
    input.release().unwrap();
}
