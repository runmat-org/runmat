use super::*;
use crate::builtins::common::test_support;
use futures::executor::block_on;
use runmat_value::{
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, SymbolicArray, SymbolicExpr,
    Tensor,
};

async fn jsonencode_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    value: Value,
    options: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let arguments = crate::call::BuiltinArguments::from_borrowed(&options, runtime.as_ref())?;
    super::jsonencode_builtin(runtime, value, arguments).await
}

fn as_string(value: Value) -> String {
    match value {
        Value::CharArray(ca) => {
            runmat_value::decode_utf16_code_units(ca.data()).expect("valid UTF-16 fixture")
        }
        Value::String(s) => s.to_string(),
        other => panic!("expected char array, got {:?}", other),
    }
}

fn error_message(err: crate::RuntimeError) -> String {
    err.message().to_string()
}

mod basic;
mod extended;
