use super::*;
#[cfg(feature = "wgpu")]
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{CharArray, StringArray, Tensor};

fn run(value: Value) -> bool {
    match block_on(isscalar_builtin(value)).expect("isscalar") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn classifies_scalars_vectors_empty_and_text_shapes() {
    assert!(run(Value::Num(5.0)));
    assert!(run(Value::Complex(2.0, -3.0)));
    assert!(!run(Value::Tensor(
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap()
    )));
    assert!(run(Value::Tensor(
        Tensor::new(vec![1.0], vec![1, 1, 1]).unwrap()
    )));
    assert!(!run(Value::Tensor(Tensor::zeros(vec![0, 1]))));
    assert!(run(Value::String("".into())));
    assert!(run(Value::StringArray(
        StringArray::new(vec!["RunMat".into()], vec![1, 1]).unwrap()
    )));
    assert!(run(Value::CharArray(
        CharArray::new(vec!['a'], 1, 1).unwrap()
    )));
    assert!(!run(Value::CharArray(CharArray::new_row("RunMat"))));
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(isscalar_builtin(Value::Num(1.0))).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:isscalar:TooManyOutputs"));
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_reads_shape_without_payload_transfer() {
    let _state = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default()).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let handle =
        gpu_helpers::upload_tensor(provider, &Tensor::new(vec![1.0], vec![1, 1]).unwrap()).unwrap();
    assert!(run(Value::GpuTensor(handle.clone())));
    provider.free(&handle).ok();
}
