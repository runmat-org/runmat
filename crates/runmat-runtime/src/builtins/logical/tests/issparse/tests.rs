use super::*;
use crate::builtins::common::{gpu_helpers, test_support};
use futures::executor::block_on;
use runmat_value::{SparseTensor, Tensor};

fn run(value: Value) -> bool {
    match block_on(issparse_builtin(value)).expect("issparse") {
        Value::Bool(value) => value,
        other => panic!("expected logical scalar, got {other:?}"),
    }
}

#[test]
fn reports_storage_representation_not_zero_count() {
    assert!(run(Value::SparseTensor(SparseTensor::zeros(3, 2))));
    assert!(!run(Value::Tensor(Tensor::zeros(vec![3, 2]))));
    assert!(!run(Value::Num(0.0)));
}

#[test]
fn resident_dense_values_are_not_sparse() {
    test_support::with_test_provider(|provider| {
        let handle = gpu_helpers::upload_tensor(provider, &Tensor::zeros(vec![2, 2])).unwrap();
        assert!(!run(Value::GpuTensor(handle.clone())));
        provider.free(&handle).ok();
    });
}

#[test]
fn rejects_excess_outputs_with_catalog_error() {
    let _outputs = crate::output_count::push_output_count(Some(2));
    let error = block_on(issparse_builtin(Value::Num(1.0))).expect_err("two outputs");
    assert_eq!(error.identifier(), Some("RunMat:issparse:TooManyOutputs"));
}
