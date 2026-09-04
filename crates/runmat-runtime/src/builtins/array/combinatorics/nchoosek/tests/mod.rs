mod exact;
mod host;

use super::*;
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
