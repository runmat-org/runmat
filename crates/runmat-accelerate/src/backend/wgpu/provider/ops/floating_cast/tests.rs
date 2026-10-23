use super::*;
#[path = "tests/lifecycle.rs"]
mod lifecycle;
#[path = "tests/views.rs"]
mod views;
use futures::executor::block_on;
use runmat_accelerate_api::{
    HostLogicalTensorView, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
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

#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; run explicitly in the F64 qualification gate"]
fn postallocation_unsupported_cast_fault_is_terminal_for_both_target_precisions() {
    let provider = provider();
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    for (data, target) in [
        (HostNumericDataView::F32(&[1.25_f32]), NumericPrecision::F64),
        (HostNumericDataView::F64(&[1.25_f64]), NumericPrecision::F32),
    ] {
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data,
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let records_before = provider.allocations.record_count().unwrap();
        let mut produced_buffer = None;
        let error = block_on(provider.floating_cast_exec_typed_after_allocation(
            &input,
            target,
            |output| {
                produced_buffer = Some(weak_buffer(output));
                Err(runmat_accelerate_api::unsupported_provider_operation(
                    "fault after floating cast output allocation",
                ))
            },
        ))
        .expect_err("postallocation fault must remain terminal");
        assert!(produced_buffer
            .expect("checkpoint observed the physical output")
            .upgrade()
            .is_none());
        assert_eq!(provider.allocations.record_count().unwrap(), records_before);
        assert!(!error.is_unsupported());
        assert!(error.source().is_some_and(|source| {
            source.is::<runmat_accelerate_api::UnsupportedProviderOperation>()
        }));
        input.allocation_lease().validate_live().unwrap();
        input.release().unwrap();
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires an actual F32 WGPU adapter; run explicitly in the device qualification gate"]
fn logical_f32_copy_marks_a_postallocation_failure_without_widening() {
    let provider = provider();
    block_on(verify_logical_f32_copy_and_postallocation_failure(provider));
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_logical_f32_copy_drops_unpublished_output_after_typed_fault() {
    let provider =
        WgpuProvider::new_async(crate::backend::wgpu::provider::WgpuProviderOptions::default())
            .await
            .expect("this browser qualification requires a real WebGPU adapter/device");
    verify_logical_f32_copy_and_postallocation_failure(&provider).await;
}

async fn verify_logical_f32_copy_and_postallocation_failure(provider: &WgpuProvider) {
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let input = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[0, 1],
            shape: &[1, 2],
        })
        .unwrap();
    assert_eq!(
        runmat_accelerate_api::handle_precision(&input),
        Some(ProviderPrecision::F32)
    );
    let numeric = provider
        .unary_single(&input)
        .await
        .expect("an authentic logical F32 producer converts to a numeric F32 allocation");
    assert_eq!(numeric.shape(), &[1, 2]);
    assert_eq!(
        numeric.descriptor().element_type,
        Some(NumericElementType::F32)
    );
    assert_eq!(numeric.descriptor().storage, Some(GpuTensorStorage::Real));
    assert!(!runmat_accelerate_api::handle_is_logical(&numeric));
    assert!(!numeric.shares_allocation_with(&input));
    assert_eq!(
        provider.download_numeric(&numeric).await.unwrap().data,
        HostNumericDataOwned::F32(vec![0.0, 1.0])
    );
    numeric.release().unwrap();
    input.allocation_lease().validate_live().unwrap();
    let ordinary = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[0.0, 1.0]),
            shape: &[1, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(ordinary
        .with_view(ordinary.view().clone().with_logical(true))
        .is_err());
    ordinary.release().unwrap();
    let records_before = provider.allocations.record_count().unwrap();
    let mut produced_buffer = None;
    let error = provider
        .floating_cast_exec_typed_after_allocation(&input, NumericPrecision::F32, |output| {
            produced_buffer = Some(weak_buffer(output));
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "fault after logical F32 copy allocation",
            ))
        })
        .await
        .expect_err("a logical allocation requires a physical numeric copy");
    assert!(produced_buffer
        .expect("checkpoint observed the physical output")
        .upgrade()
        .is_none());
    assert_eq!(provider.allocations.record_count().unwrap(), records_before);
    assert!(!error.is_unsupported());
    assert!(error
        .source()
        .is_some_and(|source| source.is::<runmat_accelerate_api::UnsupportedProviderOperation>()));
    input.allocation_lease().validate_live().unwrap();
    input.release().unwrap();
}

fn provider() -> &'static WgpuProvider {
    crate::backend::wgpu::provider::register_wgpu_provider(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU adapter is required for typed floating-cast qualification")
}

#[test]
fn actual_wgpu_single_buffers_preserve_f32_empty_nd_and_independent_handle_lifetimes() {
    let provider = provider();
    for shape in [vec![0, 2, 3], vec![1, 1, 2]] {
        let values = vec![1.25_f32; shape.iter().product()];
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&values),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let output = block_on(provider.unary_single(&input)).unwrap();
        assert!(output.shares_allocation_with(&input));
        assert_eq!(output.shape(), shape);
        assert_eq!(
            output.descriptor().element_type,
            Some(NumericElementType::F32)
        );
        assert_eq!(
            provider.get_entry_raw(&output).unwrap().allocated_bytes,
            if values.is_empty() {
                provider.element_size as u64
            } else {
                values.len() as u64 * 4
            }
        );
        input.release().unwrap();
        assert_eq!(
            block_on(provider.download_numeric(&output)).unwrap().data,
            HostNumericDataOwned::F32(values)
        );
        output.release().unwrap();
    }
}

#[test]
#[ignore = "requires an actual shader-f64 WGPU adapter; run explicitly in the F64 qualification gate"]
fn actual_wgpu_f64_to_f32_uses_target_sized_storage_and_preserves_input() {
    let provider = provider();
    assert_eq!(
        provider.precision(),
        ProviderPrecision::F64,
        "F64 gate requires shader-f64"
    );
    for shape in [vec![0, 2, 3], vec![1, 1, 6]] {
        let data = if shape[0] == 0 {
            vec![]
        } else {
            vec![1.25, -2.5, 16_777_217.0, f64::MAX, f64::INFINITY, f64::NAN]
        };
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F64(&data),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let output = block_on(provider.unary_single(&input)).unwrap();
        assert_eq!(output.shape(), shape);
        assert_eq!(
            output.descriptor().element_type,
            Some(NumericElementType::F32)
        );
        assert_eq!(
            provider.get_entry_raw(&output).unwrap().allocated_bytes,
            (data.len() as u64 * 4).max(4)
        );
        let HostNumericDataOwned::F32(actual) =
            block_on(provider.download_numeric(&output)).unwrap().data
        else {
            panic!("physical F32 output");
        };
        for (actual, input) in actual.iter().zip(&data) {
            let expected = *input as f32;
            assert!(
                actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan())
            );
        }
        output.release().unwrap();
        assert!(block_on(provider.download_numeric(&input)).is_ok());
        let duplicate = block_on(provider.unary_double(&input)).unwrap();
        input.release().unwrap();
        assert_eq!(
            duplicate.descriptor().element_type,
            Some(NumericElementType::F64)
        );
        assert!(block_on(provider.download_numeric(&duplicate)).is_ok());
        duplicate.release().unwrap();
    }
}
