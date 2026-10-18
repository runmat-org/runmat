mod exact;
mod host;

use super::*;

#[test]
fn requested_outputs_admit_zero_or_one_and_reject_excess_before_provider_access() {
    for count in [None, Some(0), Some(1)] {
        let _outputs = crate::output_count::push_output_count(count).expect("output-count scope");
        call(Value::Num(5.0), Value::Num(2.0)).expect("supported output request");
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
    let error =
        call(resident, Value::Num(2.0)).expect_err("output admission precedes provider lookup");
    assert_eq!(error.identifier(), Some("RunMat:nchoosek:TooManyOutputs"));
}
use crate::BuiltinResult;
use futures::executor::block_on;
use runmat_value::{IntValue, Tensor, Value};

fn call(first: Value, selection: Value) -> BuiltinResult<Value> {
    block_on(nchoosek_builtin(first, selection))
}

fn one_like(value: &IntValue) -> IntValue {
    match value {
        IntValue::I8(_) => IntValue::I8(1),
        IntValue::I16(_) => IntValue::I16(1),
        IntValue::I32(_) => IntValue::I32(1),
        IntValue::I64(_) => IntValue::I64(1),
        IntValue::U8(_) => IntValue::U8(1),
        IntValue::U16(_) => IntValue::U16(1),
        IntValue::U32(_) => IntValue::U32(1),
        IntValue::U64(_) => IntValue::U64(1),
    }
}
