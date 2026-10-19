use super::*;
use crate::simple_provider::InProcessProvider;
use futures::executor::block_on;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
    NumericElementType,
};
use std::sync::atomic::Ordering;

fn upload_f32(provider: &InProcessProvider, data: &[f32], shape: &[usize]) -> GpuTensorHandle {
    provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F32(data),
            shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

fn upload_f64(provider: &InProcessProvider, data: &[f64], shape: &[usize]) -> GpuTensorHandle {
    provider
        .upload_numeric(&HostNumericTensorView {
            data: HostNumericDataView::F64(data),
            shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

#[test]
fn rectangular_native_f32_and_f64_preserve_class_and_ownership_without_download() {
    let provider = InProcessProvider::new();
    let left_f32 = upload_f32(&provider, &[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let right_f32 = upload_f32(&provider, &[7., 8., 9., 10., 11., 12.], &[3, 2]);
    provider.reset_telemetry();
    let output = block_on(provider.matmul(&left_f32, &right_f32)).unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    assert_eq!(output.physical().element_type(), NumericElementType::F32);
    assert_ne!(output.allocation_identity(), left_f32.allocation_identity());
    assert_ne!(
        output.allocation_identity(),
        right_f32.allocation_identity()
    );
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(
        block_on(provider.download_numeric(&output)).unwrap().data,
        HostNumericDataOwned::F32(vec![76., 100., 103., 136.])
    );
    assert!(left_f32.allocation_lease().validate_live().is_ok());
    assert!(right_f32.allocation_lease().validate_live().is_ok());

    let left_f64 = upload_f64(&provider, &[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let right_f64 = upload_f64(&provider, &[7., 8., 9., 10., 11., 12.], &[3, 2]);
    provider.reset_telemetry();
    let output = block_on(provider.matmul(&left_f64, &right_f64)).unwrap();
    assert_eq!(output.physical().element_type(), NumericElementType::F64);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert_eq!(
        block_on(provider.download_numeric(&output)).unwrap().data,
        HostNumericDataOwned::F64(vec![76., 100., 103., 136.])
    );
}

#[test]
fn materialized_transpose_and_contiguous_reshape_use_logical_geometry() {
    let provider = InProcessProvider::new();
    let base = upload_f32(&provider, &[1., 2., 3., 4., 5., 6.], &[2, 3]);
    let transpose = provider.transpose(&base).unwrap();
    let right = upload_f32(&provider, &[1., 2., 3., 4.], &[2, 2]);
    let result = block_on(provider.matmul(&transpose, &right)).unwrap();
    assert_eq!(result.shape(), &[3, 2]);
    assert_eq!(
        block_on(provider.download_numeric(&result)).unwrap().data,
        HostNumericDataOwned::F32(vec![5., 11., 17., 11., 25., 39.])
    );

    let reshape = provider.reshape(&base, &[3, 2]).unwrap();
    assert_eq!(reshape.allocation_identity(), base.allocation_identity());
    let column = upload_f32(&provider, &[2., 3.], &[2, 1]);
    let result = block_on(provider.matmul(&reshape, &column)).unwrap();
    assert_eq!(result.shape(), &[3, 1]);
    assert_eq!(
        block_on(provider.download_numeric(&result)).unwrap().data,
        HostNumericDataOwned::F32(vec![14., 19., 24.])
    );
}

#[test]
fn rejects_foreign_and_unsupported_inputs_without_consuming_live_operands() {
    let provider = InProcessProvider::new();
    let valid = upload_f32(&provider, &[1., 2.], &[1, 2]);
    let foreign_provider = InProcessProvider::new();
    let foreign = upload_f32(&foreign_provider, &[3., 4.], &[2, 1]);
    let error = block_on(provider.matmul(&valid, &foreign)).unwrap_err();
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &error
    ));
    let mixed = upload_f64(&provider, &[3., 4.], &[2, 1]);
    let error = block_on(provider.matmul(&valid, &mixed)).unwrap_err();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &error
    ));
    assert!(block_on(provider.matmul(&valid, &valid)).is_err());
    assert!(valid.allocation_lease().validate_live().is_ok());
    assert!(foreign.allocation_lease().validate_live().is_ok());
}

#[test]
fn empty_and_overflow_geometry_are_checked_before_allocation() {
    let provider = InProcessProvider::new();
    let left = upload_f32(&provider, &[], &[0, 3]);
    let right = upload_f32(&provider, &[1., 2., 3., 4., 5., 6.], &[3, 2]);
    let result = block_on(provider.matmul(&left, &right)).unwrap();
    assert_eq!(result.shape(), &[0, 2]);
    assert_eq!(
        block_on(provider.download_numeric(&result)).unwrap().data,
        HostNumericDataOwned::F32(Vec::new())
    );
    assert!(Geometry::new(&[usize::MAX, 2], &[2, 1]).is_err());
    assert!(Geometry::new(&[2, 2], &[2, usize::MAX]).is_err());
}

#[test]
fn unpolled_future_cannot_publish_an_output_or_download_inputs() {
    let provider = InProcessProvider::new();
    let left = upload_f64(&provider, &[2.], &[1, 1]);
    let right = upload_f64(&provider, &[3.], &[1, 1]);
    provider.reset_telemetry();
    let next_id = provider.next_id.load(Ordering::Relaxed);
    drop(provider.matmul(&left, &right));
    assert_eq!(provider.next_id.load(Ordering::Relaxed), next_id);
    assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    assert!(left.allocation_lease().validate_live().is_ok());
    assert!(right.allocation_lease().validate_live().is_ok());
}

#[test]
fn empty_operands_cannot_trigger_an_infallible_huge_output_allocation() {
    let geometry = Geometry::new(&[usize::MAX, 0], &[0, 1]).unwrap();
    let error = multiply_typed::<f64>(&[], &[], geometry).unwrap_err();
    assert!(error.to_string().contains("output allocation failed"));
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &error
    ));
}
