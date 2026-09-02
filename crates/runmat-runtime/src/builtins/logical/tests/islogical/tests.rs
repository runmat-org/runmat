use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{LogicalArray, SparseTensor, Tensor};

fn run(value: Value) -> bool {
    match block_on(islogical_builtin(value)).expect("islogical") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn recognizes_dense_and_sparse_logical_storage() {
    assert!(run(Value::Bool(true)));
    assert!(run(Value::LogicalArray(
        LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()
    )));
    let sparse = SparseTensor::new_logical(2, 2, vec![0, 1, 1], vec![0]).unwrap();
    assert!(run(Value::SparseTensor(sparse)));
    assert!(!run(Value::Tensor(
        Tensor::new(vec![1.0], vec![1, 1]).unwrap()
    )));
}

#[test]
fn resident_metadata_distinguishes_numeric_and_logical_classes() {
    test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
        let handle = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
        assert!(!run(Value::GpuTensor(handle.clone())));
        assert!(run(gpu_helpers::logical_gpu_value(handle.clone())));
        provider.free(&handle).ok();
    });
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(islogical_builtin(Value::Bool(true))).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:islogical:TooManyOutputs"));
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_metadata_matches_gathered_class() {
    let _guard = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default()).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let tensor = Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap();
    let handle = gpu_helpers::upload_tensor(provider, &tensor).unwrap();
    assert!(!run(Value::GpuTensor(handle.clone())));
    assert!(run(gpu_helpers::logical_gpu_value(handle.clone())));
    provider.free(&handle).ok();
}
