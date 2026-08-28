use runmat_types::OperatorKind;
use runmat_value::{LogicalArray, Value};

use crate::runtime_error::semantic_error;
use crate::RuntimeError;

pub async fn identity(operator: OperatorKind, prototype: &Value) -> Result<Value, RuntimeError> {
    match operator {
        OperatorKind::Add | OperatorKind::Subtract => like_constructor("zeros", prototype).await,
        OperatorKind::ElementwiseMultiply => like_constructor("ones", prototype).await,
        OperatorKind::MatrixMultiply => matrix_identity(prototype).await,
        OperatorKind::ElementwiseAnd => logical_identity(prototype, true),
        OperatorKind::ElementwiseOr => logical_identity(prototype, false),
        _ => Err(unsupported(operator)),
    }
}

pub async fn combine(
    operator: OperatorKind,
    accumulator: Value,
    contribution: Value,
) -> Result<Value, RuntimeError> {
    let name = match operator {
        OperatorKind::Add | OperatorKind::Subtract => "plus",
        OperatorKind::ElementwiseMultiply => "times",
        OperatorKind::MatrixMultiply => "mtimes",
        OperatorKind::ElementwiseAnd => "and",
        OperatorKind::ElementwiseOr => "or",
        _ => return Err(unsupported(operator)),
    };
    crate::call_builtin_async(name, &[accumulator, contribution]).await
}

async fn like_constructor(name: &str, prototype: &Value) -> Result<Value, RuntimeError> {
    crate::call_builtin_async(name, &[Value::String("like".into()), prototype.clone()]).await
}

async fn matrix_identity(prototype: &Value) -> Result<Value, RuntimeError> {
    let shape =
        numeric_shape(prototype).ok_or_else(|| unsupported(OperatorKind::MatrixMultiply))?;
    let rows = shape.first().copied().unwrap_or(1);
    let columns = shape.get(1).copied().unwrap_or(1);
    if rows != columns {
        return Err(semantic_error(
            "ParallelReductionShape",
            "matrix-product reduction requires a square accumulator",
        ));
    }
    crate::call_builtin_async(
        "eye",
        &[
            Value::Int(runmat_value::IntValue::U64(u64::try_from(rows).map_err(
                |_| {
                    semantic_error(
                        "ParallelReductionShape",
                        "matrix-product reduction shape exceeds its portable representation",
                    )
                },
            )?)),
            Value::String("like".into()),
            prototype.clone(),
        ],
    )
    .await
}

fn logical_identity(prototype: &Value, value: bool) -> Result<Value, RuntimeError> {
    match prototype {
        Value::Bool(_) => Ok(Value::Bool(value)),
        Value::LogicalArray(array) => {
            LogicalArray::new(vec![u8::from(value); array.data.len()], array.shape.clone())
                .map(Value::LogicalArray)
                .map_err(|error| semantic_error("ParallelReductionShape", error))
        }
        _ => Err(unsupported(if value {
            OperatorKind::ElementwiseAnd
        } else {
            OperatorKind::ElementwiseOr
        })),
    }
}

fn numeric_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Num(_) | Value::Int(_) | Value::Complex(_, _) => Some(vec![1, 1]),
        Value::Tensor(value) => Some(value.shape.clone()),
        Value::ComplexTensor(value) => Some(value.shape.clone()),
        _ => None,
    }
}

fn unsupported(operator: OperatorKind) -> RuntimeError {
    semantic_error(
        "ParallelReductionType",
        format!("parallel reduction operator {operator:?} does not support this accumulator type"),
    )
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use runmat_value::IntValue;

    use super::*;

    #[test]
    fn reduction_identities_and_combination_preserve_exact_classes() {
        let prototype = Value::Int(IntValue::U64(u64::MAX));
        assert_eq!(
            block_on(identity(OperatorKind::Add, &prototype)).expect("integer zero"),
            Value::Int(IntValue::U64(0))
        );
        assert_eq!(
            block_on(identity(OperatorKind::ElementwiseMultiply, &prototype)).expect("integer one"),
            Value::Int(IntValue::U64(1))
        );
        assert_eq!(
            block_on(combine(
                OperatorKind::Add,
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Int(IntValue::U64(1)),
            ))
            .expect("saturating integer reduction"),
            Value::Int(IntValue::U64(u64::MAX))
        );
        assert_eq!(
            block_on(combine(
                OperatorKind::Subtract,
                Value::Num(10.0),
                Value::Num(-3.0),
            ))
            .expect("ordered subtraction contribution"),
            Value::Num(7.0)
        );
    }

    #[test]
    fn logical_reduction_identity_matches_the_accumulator_shape() {
        let prototype = Value::LogicalArray(
            LogicalArray::new(vec![0, 1, 0, 1], vec![2, 2]).expect("logical prototype"),
        );
        let identity =
            block_on(identity(OperatorKind::ElementwiseAnd, &prototype)).expect("logical identity");
        assert!(matches!(
            identity,
            Value::LogicalArray(value)
                if value.shape == vec![2, 2] && value.data.as_slice() == &[1, 1, 1, 1]
        ));
    }
}
