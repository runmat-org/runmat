mod errors;
mod identity;
mod operands;

#[cfg(test)]
mod tests;

use runmat_builtins::RelationalOperator;
use runmat_value::Value;

use crate::builtins::common::gpu_helpers;
use crate::builtins::logical::rel::integer_comparison::{
    restore_explicit_comparison_result, select_comparison_output_source,
    try_complex_integer_equality_comparison, try_complex_ordering_comparison,
    try_gpu_equality_comparison, try_gpu_ordering_comparison, try_integer_comparison,
    try_real_ordering_comparison, IntegerComparisonError, IntegerComparisonOp,
};
use crate::builtins::math::symbolic::{
    symbolic_binary_broadcast, symbolic_named_binary_broadcast, SymbolicBinaryOp,
};

use errors::{runtime_error, ComparisonError};

pub(super) async fn evaluate(
    lhs: Value,
    rhs: Value,
    operator: RelationalOperator,
) -> crate::BuiltinResult<Value> {
    let output_source = select_comparison_output_source(&lhs, &rhs, operator.name())?;
    if let (Value::GpuTensor(lhs), Value::GpuTensor(rhs)) = (&lhs, &rhs) {
        if let Some(result) = compare_resident(lhs, rhs, operator).await {
            return result;
        }
    }

    let lhs = gather(lhs, operator).await?;
    let rhs = gather(rhs, operator).await?;
    let result = evaluate_host(lhs, rhs, operator)?;
    restore_explicit_comparison_result(result, output_source.as_ref(), operator.name())
}

async fn compare_resident(
    lhs: &runmat_accelerate_api::GpuTensorHandle,
    rhs: &runmat_accelerate_api::GpuTensorHandle,
    operator: RelationalOperator,
) -> Option<crate::BuiltinResult<Value>> {
    let operation = integer_operator(operator);
    if operator.is_equality() {
        try_gpu_equality_comparison(lhs, rhs, operation).await
    } else {
        try_gpu_ordering_comparison(lhs, rhs, operation).await
    }
}

async fn gather(value: Value, operator: RelationalOperator) -> crate::BuiltinResult<Value> {
    if matches!(value, Value::GpuTensor(_)) {
        gpu_helpers::gather_value_async(&value)
            .await
            .map_err(|_| runtime_error(operator, ComparisonError::InvalidInput))
    } else {
        Ok(value)
    }
}

pub(super) fn evaluate_host(
    lhs: Value,
    rhs: Value,
    operator: RelationalOperator,
) -> crate::BuiltinResult<Value> {
    if let Some(result) = identity::compare(&lhs, &rhs, operator) {
        return Ok(result);
    }
    if let Some(result) = compare_symbolic(&lhs, &rhs, operator)? {
        return Ok(result);
    }
    if let Some(result) = compare_categorical(&lhs, &rhs, operator) {
        return result;
    }

    let (lhs, rhs) = normalize_text(lhs, rhs);
    if let Some(result) = compare_exact_numeric(&lhs, &rhs, operator)? {
        return Ok(result);
    }
    let lhs = operands::Operand::from_value(lhs, operator)?;
    let rhs = operands::Operand::from_value(rhs, operator)?;
    operands::compare(lhs, rhs, operator)
}

fn compare_symbolic(
    lhs: &Value,
    rhs: &Value,
    operator: RelationalOperator,
) -> crate::BuiltinResult<Option<Value>> {
    let result = match operator {
        RelationalOperator::Equal => symbolic_binary_broadcast(lhs, rhs, SymbolicBinaryOp::Eq),
        _ => symbolic_named_binary_broadcast(lhs, rhs, operator.name()),
    };
    result.map_err(|_| runtime_error(operator, ComparisonError::SizeMismatch))
}

fn compare_categorical(
    lhs: &Value,
    rhs: &Value,
    operator: RelationalOperator,
) -> Option<crate::BuiltinResult<Value>> {
    use crate::builtins::table::CategoricalComparison;
    let operation = match operator {
        RelationalOperator::Equal => CategoricalComparison::Eq,
        RelationalOperator::NotEqual => CategoricalComparison::Ne,
        RelationalOperator::LessThan => CategoricalComparison::Lt,
        RelationalOperator::LessThanOrEqual => CategoricalComparison::Le,
        RelationalOperator::GreaterThan => CategoricalComparison::Gt,
        RelationalOperator::GreaterThanOrEqual => CategoricalComparison::Ge,
    };
    crate::builtins::table::categorical_compare(lhs, rhs, operation)
}

fn compare_exact_numeric(
    lhs: &Value,
    rhs: &Value,
    operator: RelationalOperator,
) -> crate::BuiltinResult<Option<Value>> {
    let operation = integer_operator(operator);
    let result = if operator.is_equality() {
        try_complex_integer_equality_comparison(lhs, rhs, operation)
            .map_err(|error| runtime_error(operator, ComparisonError::from(error)))?
            .or(try_integer_comparison(lhs, rhs, operation)
                .map_err(|error| runtime_error(operator, ComparisonError::from(error)))?)
    } else {
        try_real_ordering_comparison(lhs, rhs, operation)
            .map_err(|error| runtime_error(operator, ComparisonError::from(error)))?
            .or(try_complex_ordering_comparison(lhs, rhs, operation)
                .map_err(|error| runtime_error(operator, ComparisonError::from(error)))?)
    };
    Ok(result)
}

impl From<IntegerComparisonError> for ComparisonError {
    fn from(value: IntegerComparisonError) -> Self {
        match value {
            IntegerComparisonError::SizeMismatch => Self::SizeMismatch,
            IntegerComparisonError::Internal => Self::InvalidInput,
        }
    }
}

fn integer_operator(operator: RelationalOperator) -> IntegerComparisonOp {
    match operator {
        RelationalOperator::Equal => IntegerComparisonOp::Eq,
        RelationalOperator::NotEqual => IntegerComparisonOp::Ne,
        RelationalOperator::LessThan => IntegerComparisonOp::Lt,
        RelationalOperator::LessThanOrEqual => IntegerComparisonOp::Le,
        RelationalOperator::GreaterThan => IntegerComparisonOp::Gt,
        RelationalOperator::GreaterThanOrEqual => IntegerComparisonOp::Ge,
    }
}

fn normalize_text(lhs: Value, rhs: Value) -> (Value, Value) {
    match (lhs, rhs) {
        (Value::CharArray(chars), Value::String(text)) => (
            Value::String(chars.data.into_iter().collect()),
            Value::String(text),
        ),
        (Value::String(text), Value::CharArray(chars)) => (
            Value::String(text),
            Value::String(chars.data.into_iter().collect()),
        ),
        (Value::CharArray(chars), Value::StringArray(text)) => (
            Value::String(chars.data.into_iter().collect()),
            Value::StringArray(text),
        ),
        (Value::StringArray(text), Value::CharArray(chars)) => (
            Value::StringArray(text),
            Value::String(chars.data.into_iter().collect()),
        ),
        values => values,
    }
}
