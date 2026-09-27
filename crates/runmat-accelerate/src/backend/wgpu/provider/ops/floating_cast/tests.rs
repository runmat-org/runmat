use super::*;
#[path = "tests/views.rs"]
mod views;
use futures::executor::block_on;
use runmat_accelerate_api::{HostNumericDataOwned, HostNumericDataView, HostNumericTensorView};

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
        assert_ne!(output.buffer_id, input.buffer_id);
        assert_eq!(output.shape, shape);
        assert_eq!(
            output.descriptor.element_type,
            Some(NumericElementType::F32)
        );
        assert_eq!(
            provider.get_entry_raw(&output).unwrap().allocated_bytes,
            values.len() as u64 * 4
        );
        provider.free(&input).unwrap();
        assert_eq!(
            block_on(provider.download_numeric(&output)).unwrap().data,
            HostNumericDataOwned::F32(values)
        );
        provider.free(&output).unwrap();
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
        assert_eq!(output.shape, shape);
        assert_eq!(
            output.descriptor.element_type,
            Some(NumericElementType::F32)
        );
        assert_eq!(
            provider.get_entry_raw(&output).unwrap().allocated_bytes,
            data.len() as u64 * 4
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
        provider.free(&output).unwrap();
        assert!(block_on(provider.download_numeric(&input)).is_ok());
        let duplicate = block_on(provider.unary_double(&input)).unwrap();
        provider.free(&input).unwrap();
        assert_eq!(
            duplicate.descriptor.element_type,
            Some(NumericElementType::F64)
        );
        assert!(block_on(provider.download_numeric(&duplicate)).is_ok());
        provider.free(&duplicate).unwrap();
    }
}
