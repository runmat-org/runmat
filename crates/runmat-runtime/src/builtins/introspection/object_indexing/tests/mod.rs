use super::*;
use futures::executor::block_on;
use runmat_types::MemberAccess;
use runmat_value::{CellArray, IntValue, ObjectInstance, StructArray, StructValue, Tensor};
use std::collections::HashMap;

pub(super) fn substruct(kind: &str, subs: Value) -> Value {
    let mut st = StructValue::new();
    st.insert("type", Value::String(kind.to_string()));
    st.insert("subs", subs);
    Value::Struct(st)
}

pub(super) fn cell(values: Vec<Value>) -> Value {
    let cols = values.len();
    Value::Cell(CellArray::new(values, 1, cols).expect("cell"))
}

pub(super) fn call(target: Value, subscript: Value, context: Value) -> Value {
    block_on(num_arguments_from_subscript_builtin(
        target, subscript, context,
    ))
    .expect("numArgumentsFromSubscript")
}

pub(super) fn call_err(target: Value, subscript: Value, context: Value) -> crate::RuntimeError {
    block_on(num_arguments_from_subscript_builtin(
        target, subscript, context,
    ))
    .expect_err("numArgumentsFromSubscript should fail")
}

mod default_indexing;
mod cardinality;
mod validation;
mod cutover;
mod shape_cardinality;
mod overload_outputs;
