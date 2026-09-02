use super::*;
#[cfg(feature = "wgpu")]
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::Tensor;

fn run(value: Value) -> bool {
    match block_on(iscolumn_builtin(value)).expect("iscolumn") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn classifies_column_scalar_empty_and_higher_rank_geometry() {
    assert!(run(Value::Num(1.0)));
    assert!(!run(Value::Tensor(
        Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()
    )));
    assert!(run(Value::Tensor(
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()
    )));
    assert!(run(Value::Tensor(Tensor::zeros(vec![0, 1]))));
    assert!(!run(Value::Tensor(Tensor::zeros(vec![1, 0]))));
    assert!(run(Value::Tensor(
        Tensor::new(vec![1.0, 2.0], vec![2, 1, 1]).unwrap()
    )));
    assert!(!run(Value::Tensor(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]).unwrap()
    )));
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_reads_shape_without_payload_transfer() {
    let _state = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default()).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let handle = gpu_helpers::upload_tensor(
        provider,
        &Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap(),
    )
    .unwrap();
    assert!(run(Value::GpuTensor(handle.clone())));
    provider.free(&handle).ok();
}
