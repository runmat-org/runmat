//! Canonical scalar-argument application of a symbolic function reference.

use runmat_value::{SymbolicExpr, Value};

use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub fn scalar_from_value(value: &Value) -> Result<SymbolicExpr, RuntimeError> {
    match value {
        Value::Symbolic(expr) => Ok(expr.clone()),
        Value::Num(value) => Ok(SymbolicExpr::constant(*value)),
        Value::Int(value) => Ok(SymbolicExpr::constant(value.to_f64())),
        Value::Bool(value) => Ok(SymbolicExpr::constant(if *value { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) if crate::builtins::common::tensor::is_scalar_tensor(tensor) => Ok(
            SymbolicExpr::constant(crate::builtins::common::tensor::tensor_value_f64(tensor, 0)),
        ),
        Value::LogicalArray(logical) if logical.data.len() == 1 => {
            Ok(SymbolicExpr::constant(if logical.data[0] != 0 {
                1.0
            } else {
                0.0
            }))
        }
        other => Err(semantic_error(
            "InvalidSymbolicFunctionArgument",
            format!("symbolic function arguments must be scalar values, got {other:?}"),
        )),
    }
}

pub fn apply_function_reference(
    name: &str,
    parameters: &[runmat_value::Utf8Text],
    arguments: &[Value],
) -> Result<Value, RuntimeError> {
    if arguments.len() != parameters.len() {
        return Err(semantic_error(
            "SymbolicFunctionArity",
            format!(
                "symbolic function {name} expects {} argument(s), got {}",
                parameters.len(),
                arguments.len()
            ),
        ));
    }
    let arguments = arguments
        .iter()
        .map(scalar_from_value)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Value::Symbolic(SymbolicExpr::function_call(
        name.to_string(),
        arguments,
    )))
}
