use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataView, HostIntegerTensorView, HostLogicalTensorView,
    HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
};

fn actual_adapter() -> &'static WgpuProvider {
    crate::backend::wgpu::provider::register_wgpu_provider(
        crate::backend::wgpu::provider::WgpuProviderOptions::default(),
    )
    .expect("actual WGPU adapter required; an unavailable adapter is not a passing test")
}

#[test]
fn wide_word_conversion_has_one_rounding_boundary() {
    fn unsigned(low: u32, high: u32) -> f64 {
        f64::from(high) * 4_294_967_296.0 + f64::from(low)
    }
    fn signed(value: i64) -> f64 {
        let bits = value as u64;
        let low = bits as u32;
        let high = (bits >> 32) as u32;
        if (high & 0x8000_0000) == 0 {
            return unsigned(low, high);
        }
        let magnitude_low = (!low).wrapping_add(1);
        let carry = u32::from(magnitude_low == 0);
        -unsigned(magnitude_low, (!high).wrapping_add(carry))
    }
    for value in [0, 1, (1_u64 << 53) + 1, 1_u64 << 63, u64::MAX] {
        assert_eq!(unsigned(value as u32, (value >> 32) as u32), value as f64);
    }
    for value in [i64::MIN, -((1_i64 << 53) + 1), -1, 0, 1, i64::MAX] {
        assert_eq!(signed(value), value as f64);
    }
}

#[test]
#[ignore = "explicit actual F64 WGPU hardware gate; run with --ignored on a shader-f64 adapter"]
fn ordinary_integer_sum_stays_resident_and_differs_from_native_saturation() {
    let provider = actual_adapter();
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let input = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I8(&[100, 100]),
            shape: &[2, 1],
        })
        .unwrap();
    let ordinary = block_on(provider.reduce_sum(&input)).unwrap();
    let native = block_on(provider.reduce_integer_sum_native(&input)).unwrap();
    assert_eq!(
        ordinary.descriptor().element_type,
        Some(NumericElementType::F64)
    );
    assert_eq!(
        block_on(provider.download_numeric(&ordinary)).unwrap().data,
        HostNumericDataOwned::F64(vec![200.0])
    );
    assert_eq!(
        block_on(provider.download_integer(&native)).unwrap().data,
        runmat_accelerate_api::HostIntegerDataOwned::I8(vec![127])
    );
    input.release().unwrap();
    ordinary.release().unwrap();
    native.release().unwrap();
}

#[test]
#[ignore = "explicit actual F64 WGPU hardware gate; run with --ignored on a shader-f64 adapter"]
fn ordinary_integer_sum_handles_wide_rounding_nd_and_lazy_transpose() {
    let provider = actual_adapter();
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let wide = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::U64(&[u64::MAX]),
            shape: &[1, 1],
        })
        .unwrap();
    let wide_sum = block_on(provider.reduce_sum(&wide)).unwrap();
    assert_eq!(
        block_on(provider.download_numeric(&wide_sum)).unwrap().data,
        HostNumericDataOwned::F64(vec![u64::MAX as f64])
    );
    wide_sum.release().unwrap();
    wide.release().unwrap();

    let input = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I16(&[1, 2, 3, 4, 5, 6, 7, 8]),
            shape: &[2, 2, 2],
        })
        .unwrap();
    let reduced = block_on(provider.reduce_sum_dim(&input, 1)).unwrap();
    assert_eq!(reduced.shape(), &[2, 1, 2]);
    assert_eq!(
        block_on(provider.download_numeric(&reduced)).unwrap().data,
        HostNumericDataOwned::F64(vec![4.0, 6.0, 12.0, 14.0])
    );
    reduced.release().unwrap();
    input.release().unwrap();

    let matrix = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I32(&[1, 2, 3, 4, 5, 6]),
            shape: &[2, 3],
        })
        .unwrap();
    let view = provider.transpose_exec(&matrix).unwrap();
    let columns = block_on(provider.reduce_sum_dim(&view, 0)).unwrap();
    assert_eq!(
        block_on(provider.download_numeric(&columns)).unwrap().data,
        HostNumericDataOwned::F64(vec![9.0, 12.0])
    );
    columns.release().unwrap();
    view.release().unwrap();
    matrix.release().unwrap();
}

#[test]
#[ignore = "explicit actual F64 WGPU hardware gate; run with --ignored on a shader-f64 adapter"]
fn ordinary_single_sum_casts_before_reduction() {
    let provider = actual_adapter();
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let input = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[16_777_216.0, 1.0, -16_777_216.0]),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let output = block_on(provider.reduce_sum(&input)).unwrap();
    assert_eq!(output.physical().element_type(), NumericElementType::F64);
    assert_eq!(
        block_on(provider.download_numeric(&output)).unwrap().data,
        HostNumericDataOwned::F64(vec![1.0])
    );
    output.release().unwrap();
    input.release().unwrap();
}

#[test]
#[ignore = "explicit actual F32 WGPU hardware gate; run with --ignored on an F32-only adapter"]
fn f32_only_device_rejects_double_integer_and_logical_sums_before_dispatch() {
    let provider = actual_adapter();
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F32
    );
    let integer = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I8(&[100, 100]),
            shape: &[2, 1],
        })
        .unwrap();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &block_on(provider.reduce_sum(&integer)).unwrap_err()
    ));
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &block_on(provider.reduce_sum_dim(&integer, 0)).unwrap_err()
    ));
    integer.release().unwrap();

    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[1, 1],
            shape: &[2, 1],
        })
        .unwrap();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &block_on(provider.reduce_sum(&logical)).unwrap_err()
    ));
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &block_on(provider.reduce_sum_dim(&logical, 0)).unwrap_err()
    ));
    logical.release().unwrap();

    let single = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[16_777_216.0, 1.0, -16_777_216.0]),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &block_on(provider.reduce_sum(&single)).unwrap_err()
    ));
    single.release().unwrap();
}
