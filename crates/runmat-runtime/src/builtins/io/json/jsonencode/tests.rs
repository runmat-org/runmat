use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, StructValue, SymbolicArray,
    SymbolicExpr, Tensor,
};

fn as_string(value: Value) -> String {
    match value {
        Value::CharArray(ca) => ca.data.iter().collect(),
        Value::String(s) => s,
        other => panic!("expected char array, got {:?}", other),
    }
}

fn error_message(err: crate::RuntimeError) -> String {
    err.message().to_string()
}

mod basic;
mod extended;
