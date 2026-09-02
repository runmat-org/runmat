use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{LogicalArray, SparseTensor, Tensor};

fn run(value: Value) -> bool {
    match block_on(isnumeric_builtin(value)).expect("isnumeric") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn recognizes_dense_complex_integer_and_sparse_numeric_storage() {
    assert!(run(Value::Num(1.0)));
    assert!(run(Value::from(u64::MAX)));
    assert!(run(Value::Complex(1.0, 2.0)));
    let sparse = SparseTensor::new(2, 2, vec![0, 1, 1], vec![0], vec![3.0]).unwrap();
    assert!(run(Value::SparseTensor(sparse)));
    let logical_sparse = SparseTensor::new_logical(2, 2, vec![0, 1, 1], vec![0]).unwrap();
    assert!(!run(Value::SparseTensor(logical_sparse)));
    assert!(!run(Value::LogicalArray(
        LogicalArray::new(vec![1], vec![1, 1]).unwrap()
    )));
}

#[test]
fn resident_metadata_distinguishes_numeric_and_logical_classes() {
    test_support::with_test_provider(|provider| {
        let handle =
            gpu_helpers::upload_tensor(provider, &Tensor::new(vec![1.0], vec![1, 1]).unwrap())
                .unwrap();
        assert!(run(Value::GpuTensor(handle.clone())));
        assert!(!run(gpu_helpers::logical_gpu_value(handle.clone())));
        provider.free(&handle).ok();
    });
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(isnumeric_builtin(Value::Num(1.0))).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:isnumeric:TooManyOutputs"));
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_metadata_matches_gathered_class() {
    let _guard = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default()).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let handle =
        gpu_helpers::upload_tensor(provider, &Tensor::new(vec![1.0], vec![1, 1]).unwrap()).unwrap();
    assert!(run(Value::GpuTensor(handle.clone())));
    assert!(!run(gpu_helpers::logical_gpu_value(handle.clone())));
    provider.free(&handle).ok();
}
