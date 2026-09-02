use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{CellArray, CharArray, StringArray, Tensor};

fn run(value: Value) -> bool {
    match block_on(isempty_builtin(value)).expect("isempty") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn classifies_numeric_text_and_cell_geometry() {
    assert!(run(Value::Tensor(Tensor::zeros(vec![0, 3]))));
    assert!(!run(Value::Num(5.0)));
    assert!(run(Value::CharArray(CharArray::new_row(""))));
    assert!(!run(Value::String(String::new())));
    assert!(run(Value::StringArray(
        StringArray::new(Vec::<String>::new(), vec![0, 2]).unwrap()
    )));
    assert!(run(Value::Cell(CellArray::new(Vec::new(), 0, 2).unwrap())));
    assert!(!run(Value::Cell(
        CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()
    )));
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(isempty_builtin(Value::Num(1.0))).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:isempty:TooManyOutputs"));
}

#[test]
fn provider_handle_shape_requires_no_payload_read() {
    test_support::with_test_provider(|provider| {
        let handle = gpu_helpers::upload_tensor(provider, &Tensor::zeros(vec![0, 4])).unwrap();
        assert!(run(Value::GpuTensor(handle.clone())));
        provider.free(&handle).ok();
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_reads_shape_without_payload_transfer() {
    let _state = test_support::accel_test_lock();
    runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(Default::default()).unwrap();
    let provider = runmat_accelerate_api::provider().unwrap();
    let _provider = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
    let handle = gpu_helpers::upload_tensor(provider, &Tensor::zeros(vec![0, 4])).unwrap();
    assert!(run(Value::GpuTensor(handle.clone())));
    provider.free(&handle).ok();
}
