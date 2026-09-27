use super::*;
use crate::builtins::common::test_support;

fn logical_fallback(provider: &'static dyn runmat_accelerate_api::AccelProvider) {
    let source = runmat_accelerate_api::upload_logical_protecting(
        provider,
        &runmat_accelerate_api::HostLogicalTensorView {
            data: &[0, 1, 0, 1, 1, 0],
            shape: &[3, 2],
        },
        &[],
    )
    .unwrap();
    let result = futures::executor::block_on(fftshift_gpu_fallback(source.clone(), &[0])).unwrap();
    let Value::GpuTensor(output) = result else {
        panic!("supported logical restoration must remain resident");
    };
    assert!(runmat_accelerate_api::handle_is_logical(&output));
    assert_eq!(output.device_id(), source.device_id());
    assert_eq!(output.shape(), &[3, 2]);
    assert!(!output.shares_allocation_with(&source));
    let gathered =
        futures::executor::block_on(gpu_helpers::gather_value_async(&Value::GpuTensor(output)))
            .unwrap();
    let Value::LogicalArray(array) = gathered else {
        panic!("logical class must survive owner-preserving restoration");
    };
    assert_eq!(array.shape, vec![3, 2]);
    assert_eq!(array.data, vec![0, 0, 1, 0, 1, 1]);
    source.allocation_lease().validate_live().unwrap();
}

#[test]
fn fftshift_logical_fallback_preserves_owner_class_and_borrowed_input() {
    test_support::with_test_provider(logical_fallback);
    test_support::with_f32_test_provider(logical_fallback);
}
