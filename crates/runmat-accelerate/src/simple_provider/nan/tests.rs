use super::*;
use futures::executor::block_on;

#[test]
fn nan_transforms_preserve_native_precision_shape_and_borrowed_input() {
    let provider = InProcessProvider::new();
    for input in [
        HostNumericDataView::F32(&[f32::NAN, -0.0, f32::INFINITY, -3.0]),
        HostNumericDataView::F64(&[f64::NAN, -0.0, f64::INFINITY, -3.0]),
    ] {
        let source = provider
            .upload_numeric(&HostNumericTensorView {
                data: input,
                shape: &[2, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        let cleaned = provider.map_nan_to_zero(&source).unwrap();
        let mask = provider.not_nan_mask(&source).unwrap();
        for output in [&cleaned, &mask] {
            assert_eq!(output.shape(), source.shape());
            assert_eq!(
                output.physical().element_type(),
                source.physical().element_type()
            );
            assert_eq!(output.device_id(), source.device_id());
            assert!(!output.view().is_logical());
            assert!(!output.shares_allocation_with(&source));
        }
        match block_on(provider.download_numeric(&cleaned)).unwrap().data {
            HostNumericDataOwned::F32(values) => {
                assert_eq!(values[0], 0.0);
                assert_eq!(values[1].to_bits(), (-0.0_f32).to_bits());
                assert_eq!(values[2], f32::INFINITY);
                assert_eq!(values[3], -3.0);
            }
            HostNumericDataOwned::F64(values) => {
                assert_eq!(values[0], 0.0);
                assert_eq!(values[1].to_bits(), (-0.0_f64).to_bits());
                assert_eq!(values[2], f64::INFINITY);
                assert_eq!(values[3], -3.0);
            }
            other => panic!("NaN replacement changed numeric class: {other:?}"),
        }
        match block_on(provider.download_numeric(&mask)).unwrap().data {
            HostNumericDataOwned::F32(values) => assert_eq!(values, [0.0, 1.0, 1.0, 1.0]),
            HostNumericDataOwned::F64(values) => assert_eq!(values, [0.0, 1.0, 1.0, 1.0]),
            other => panic!("NaN mask changed numeric class: {other:?}"),
        }
        source.allocation_lease().validate_live().unwrap();
    }
}

#[test]
fn nan_transforms_keep_empty_shapes_and_reject_nonfloating_or_foreign_inputs() {
    let provider = InProcessProvider::new();
    let empty = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[]),
            shape: &[0, 2, 3],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    for output in [
        provider.map_nan_to_zero(&empty).unwrap(),
        provider.not_nan_mask(&empty).unwrap(),
    ] {
        assert_eq!(output.shape(), &[0, 2, 3]);
        assert_eq!(output.physical().element_type(), NumericElementType::F32);
    }
    let integer = provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::I32(&[1, 2]),
            shape: &[2, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(provider
        .map_nan_to_zero(&integer)
        .unwrap_err()
        .is_unsupported());
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &provider.not_nan_mask(&integer).unwrap_err()
    ));
    let foreign = InProcessProvider::new();
    let input = foreign
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[f64::NAN]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    assert!(provider.map_nan_to_zero(&input).is_err());
    assert!(provider.not_nan_mask(&input).is_err());
}
