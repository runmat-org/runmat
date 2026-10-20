mod containers;
mod integers;
mod planning;
mod provider;

use super::*;

#[test]
fn requested_outputs_admit_zero_or_one_and_reject_excess_before_provider_access() {
    for count in [None, Some(0), Some(1)] {
        let _outputs = crate::output_count::push_output_count(count).expect("output-count scope");
        call(Value::Num(3.0), Vec::new()).expect("supported output request");
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
    let error = call(resident, Vec::new()).expect_err("output admission precedes provider lookup");
    assert_eq!(
        error.identifier(),
        Some("RunMat:combinations:TooManyOutputs")
    );
}
use crate::builtins::table::{table_height, table_variables};
use crate::BuiltinResult;
use futures::executor::block_on;
use runmat_value::{ObjectInstance, Value};

fn call(first: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    block_on(combinations_builtin(
        None,
        first,
        crate::builtins::common::test_support::builtin_arguments(rest),
    ))
}

fn table(value: Value) -> ObjectInstance {
    let Value::Object(table) = value else {
        panic!("expected table")
    };
    table
}
