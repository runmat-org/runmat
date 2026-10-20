mod host;
mod provider;

use super::*;

#[test]
fn requested_outputs_admit_zero_or_one_and_reject_excess_before_provider_access() {
    for count in [None, Some(0), Some(1)] {
        let _outputs = crate::output_count::push_output_count(count).expect("output-count scope");
        call(Value::Num(3.0)).expect("supported output request");
    }
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let allocation_owner = runmat_accelerate_api::test_support::TestAllocationOwner::new();
    let resident = Value::GpuTensor(
        allocation_owner
            .allocate_zeroed(
                vec![2, 2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .expect("genuine unregistered allocation")
            .handle,
    );
    let error = call(resident).expect_err("output admission precedes provider lookup");
    assert_eq!(error.identifier(), Some("RunMat:perms:TooManyOutputs"));
}
use crate::BuiltinResult;
use futures::executor::block_on;
use runmat_value::{ComplexTensor, Tensor, Value};

fn call(value: Value) -> BuiltinResult<Value> {
    block_on(perms_builtin(
        value,
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
}

fn tensor_rows(tensor: &Tensor) -> Vec<Vec<f64>> {
    (0..tensor.rows())
        .map(|row| {
            (0..tensor.cols())
                .map(|column| tensor.materialize_f64()[column * tensor.rows() + row])
                .collect()
        })
        .collect()
}

fn complex_rows(tensor: &ComplexTensor) -> Vec<Vec<(f64, f64)>> {
    (0..tensor.rows())
        .map(|row| {
            (0..tensor.cols())
                .map(|column| tensor.materialize_f64()[column * tensor.rows() + row])
                .collect()
        })
        .collect()
}
