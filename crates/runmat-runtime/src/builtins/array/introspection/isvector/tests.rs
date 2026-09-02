use super::*;
#[cfg(feature = "wgpu")]
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{CellArray, CharArray, Tensor};

fn run(value: Value) -> bool {
    match block_on(isvector_builtin(value)).expect("isvector") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn classifies_rows_columns_scalars_matrices_and_empty_shapes() {
    assert!(run(Value::Num(5.0)));
    assert!(run(Value::Tensor(
        Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()
    )));
    assert!(run(Value::Tensor(
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()
    )));
    assert!(!run(Value::Tensor(
        Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()
    )));
    assert!(run(Value::Tensor(Tensor::zeros(vec![1, 0]))));
    assert!(run(Value::Tensor(Tensor::zeros(vec![0, 1]))));
    assert!(!run(Value::Tensor(Tensor::zeros(vec![0, 3]))));
}

#[test]
fn ignores_trailing_singletons_and_uses_container_dimensions() {
    for shape in [vec![1, 1, 1], vec![3, 1, 1], vec![1, 3, 1, 1]] {
        let count = shape.iter().product();
        assert!(run(Value::Tensor(
            Tensor::new(vec![1.0; count], shape).unwrap()
        )));
    }
    assert!(!run(Value::Tensor(
        Tensor::new(vec![1.0; 3], vec![1, 1, 3, 1]).unwrap()
    )));
    assert!(run(Value::CharArray(CharArray::new_row("RunMat"))));
    assert!(run(Value::Cell(
        CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap()
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
