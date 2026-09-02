use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{ComplexTensor, SparseTensor, Tensor};

fn run(value: Value) -> bool {
    match block_on(isreal_builtin(value)).expect("isreal") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn distinguishes_real_and_complex_storage_across_dense_and_sparse_values() {
    assert!(run(Value::Num(1.0)));
    assert!(run(Value::Tensor(
        Tensor::new(vec![1.0], vec![1, 1]).unwrap()
    )));
    assert!(!run(Value::Complex(1.0, 0.0)));
    assert!(!run(Value::ComplexTensor(
        ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap()
    )));
    assert!(run(Value::SparseTensor(
        SparseTensor::new(2, 2, vec![0, 1, 1], vec![0], vec![1.0]).unwrap()
    )));
    assert!(!run(Value::SparseTensor(
        SparseTensor::new_complex(2, 2, vec![0, 1, 1], vec![0], vec![(1.0, 0.0)]).unwrap()
    )));
}

#[test]
fn resident_metadata_distinguishes_real_and_complex_storage() {
    test_support::with_test_provider(|provider| {
        let real =
            gpu_helpers::upload_tensor(provider, &Tensor::new(vec![1.0], vec![1, 1]).unwrap())
                .unwrap();
        assert!(run(Value::GpuTensor(real.clone())));
        provider.free(&real).ok();
    });
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(isreal_builtin(Value::Num(1.0))).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:isreal:TooManyOutputs"));
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_metadata_matches_storage_complexity() {
    let _guard = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default()).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let handle =
        gpu_helpers::upload_tensor(provider, &Tensor::new(vec![1.0], vec![1, 1]).unwrap()).unwrap();
    assert!(run(Value::GpuTensor(handle.clone())));
    provider.free(&handle).ok();
}
