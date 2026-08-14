use crate::builtins::common::tensor::{
    complex_tensor_element_len, complex_tensor_value_complex64, is_scalar_tensor, tensor_value_f64,
};
use runmat_types::{CallableFallbackPolicy, CallableIdentity};
use runmat_value::{IntValue, Value};
use serde::{Deserialize, Serialize};
use std::{future::Future, pin::Pin};

/// Executor-neutral representation of `end` arithmetic used by indexing
/// plans, bytecode, native IR, and semantic call adapters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EndExpr {
    End,
    Const(f64),
    Var(usize),
    ResolvedCall {
        identity: CallableIdentity,
        fallback_policy: CallableFallbackPolicy,
        args: Vec<EndExpr>,
    },
    Add(Box<EndExpr>, Box<EndExpr>),
    Sub(Box<EndExpr>, Box<EndExpr>),
    Mul(Box<EndExpr>, Box<EndExpr>),
    Div(Box<EndExpr>, Box<EndExpr>),
    LeftDiv(Box<EndExpr>, Box<EndExpr>),
    Pow(Box<EndExpr>, Box<EndExpr>),
    Neg(Box<EndExpr>),
    Pos(Box<EndExpr>),
    Floor(Box<EndExpr>),
    Ceil(Box<EndExpr>),
    Round(Box<EndExpr>),
    Fix(Box<EndExpr>),
}

#[derive(Debug, Clone, Copy)]
pub struct ValueToF64Error;

/// Converts an integer only when the resulting double retains its exact value.
/// `end` expressions are subsequently used as indices, so accepting a rounded
/// integer here would select a different element than the source expression.
fn exact_integer_to_f64(value: &IntValue) -> Result<f64, ValueToF64Error> {
    let converted = value.to_f64();
    let exact = match value {
        IntValue::I8(value) => converted as i128 == i128::from(*value),
        IntValue::I16(value) => converted as i128 == i128::from(*value),
        IntValue::I32(value) => converted as i128 == i128::from(*value),
        IntValue::I64(value) => converted as i128 == i128::from(*value),
        IntValue::U8(value) => converted as u128 == u128::from(*value),
        IntValue::U16(value) => converted as u128 == u128::from(*value),
        IntValue::U32(value) => converted as u128 == u128::from(*value),
        IntValue::U64(value) => converted as u128 == u128::from(*value),
    };
    exact.then_some(converted).ok_or(ValueToF64Error)
}

pub fn value_to_f64(v: &Value) -> Result<f64, ValueToF64Error> {
    match v {
        Value::Num(n) => Ok(*n),
        Value::Int(i) => exact_integer_to_f64(i),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::Tensor(t) if is_scalar_tensor(t) => match t.integer_storage() {
            Some(storage) => exact_integer_to_f64(&storage.value_at(0).ok_or(ValueToF64Error)?),
            None => Ok(tensor_value_f64(t, 0)),
        },
        Value::Complex(re, im) if im.abs() < 1e-12 => Ok(*re),
        Value::ComplexTensor(ct) if complex_tensor_element_len(ct) == 1 => {
            let value = complex_tensor_value_complex64(ct, 0);
            if value.im.abs() < 1e-12 {
                Ok(value.re)
            } else {
                Err(ValueToF64Error)
            }
        }
        _ => Err(ValueToF64Error),
    }
}

/// Resolve one context-dependent indexing expression through caller-owned
/// local storage and the canonical runtime callable path.
pub async fn resolve_end_expr_value<F>(
    dimension_length: usize,
    expression: &EndExpr,
    resolve_variable: F,
) -> Result<f64, crate::RuntimeError>
where
    F: Fn(usize) -> Option<Value> + Copy,
{
    fn evaluate<'a, F>(
        expression: &'a EndExpr,
        end_value: f64,
        resolve_variable: F,
    ) -> Pin<Box<dyn Future<Output = Result<f64, crate::RuntimeError>> + 'a>>
    where
        F: Fn(usize) -> Option<Value> + Copy + 'a,
    {
        Box::pin(async move {
            let invalid =
                |identifier, message| crate::runtime_error::semantic_error(identifier, message);
            match expression {
                EndExpr::End => Ok(end_value),
                EndExpr::Const(value) => Ok(*value),
                EndExpr::Var(local) => {
                    let mut value = resolve_variable(*local).ok_or_else(|| {
                        invalid("MissingNumericIndex", "missing variable for end expression")
                    })?;
                    if matches!(value, Value::GpuTensor(_)) {
                        value = crate::dispatcher::gather_if_needed_async(&value).await?;
                    }
                    value_to_f64(&value).map_err(|_| {
                        invalid("UnsupportedIndexType", "end expression must be numeric")
                    })
                }
                EndExpr::ResolvedCall {
                    identity,
                    fallback_policy,
                    args,
                } => {
                    let mut arguments = Vec::with_capacity(args.len());
                    for argument in args {
                        arguments.push(Value::Num(
                            evaluate(argument, end_value, resolve_variable).await?,
                        ));
                    }
                    let descriptor = crate::call::descriptor::CallableDescriptor::resolved(
                        identity.clone(),
                        arguments,
                        1,
                        *fallback_policy,
                        crate::call::descriptor::CallableCallKind::EndExpr,
                    );
                    let value =
                        crate::call::descriptor::execute_callable_descriptor(descriptor).await?;
                    let value = match value {
                        Value::OutputList(mut values) if values.len() == 1 => values.remove(0),
                        value => value,
                    };
                    value_to_f64(&value)
                        .map_err(|_| invalid("UnsupportedIndexType", "end call must return scalar"))
                }
                EndExpr::Add(left, right) => Ok(evaluate(left, end_value, resolve_variable)
                    .await?
                    + evaluate(right, end_value, resolve_variable).await?),
                EndExpr::Sub(left, right) => Ok(evaluate(left, end_value, resolve_variable)
                    .await?
                    - evaluate(right, end_value, resolve_variable).await?),
                EndExpr::Mul(left, right) => Ok(evaluate(left, end_value, resolve_variable)
                    .await?
                    * evaluate(right, end_value, resolve_variable).await?),
                EndExpr::Div(left, right) => {
                    let denominator = evaluate(right, end_value, resolve_variable).await?;
                    if denominator == 0.0 {
                        return Err(invalid("IndexOutOfBounds", "Index out of bounds"));
                    }
                    Ok(evaluate(left, end_value, resolve_variable).await? / denominator)
                }
                EndExpr::LeftDiv(left, right) => {
                    let denominator = evaluate(left, end_value, resolve_variable).await?;
                    if denominator == 0.0 {
                        return Err(invalid("IndexOutOfBounds", "Index out of bounds"));
                    }
                    Ok(evaluate(right, end_value, resolve_variable).await? / denominator)
                }
                EndExpr::Pow(left, right) => Ok(evaluate(left, end_value, resolve_variable)
                    .await?
                    .powf(evaluate(right, end_value, resolve_variable).await?)),
                EndExpr::Neg(inner) => Ok(-evaluate(inner, end_value, resolve_variable).await?),
                EndExpr::Pos(inner) => evaluate(inner, end_value, resolve_variable).await,
                EndExpr::Floor(inner) => {
                    Ok(evaluate(inner, end_value, resolve_variable).await?.floor())
                }
                EndExpr::Ceil(inner) => {
                    Ok(evaluate(inner, end_value, resolve_variable).await?.ceil())
                }
                EndExpr::Round(inner) => {
                    Ok(evaluate(inner, end_value, resolve_variable).await?.round())
                }
                EndExpr::Fix(inner) => {
                    let value = evaluate(inner, end_value, resolve_variable).await?;
                    Ok(if value >= 0.0 {
                        value.floor()
                    } else {
                        value.ceil()
                    })
                }
            }
        })
    }

    evaluate(expression, dimension_length as f64, resolve_variable).await
}

#[cfg(test)]
mod tests {
    use super::{resolve_end_expr_value, value_to_f64, EndExpr};
    use runmat_value::{IntValue, IntegerStorage, Tensor, Value};

    #[test]
    fn value_to_f64_reads_all_typed_integer_tensor_classes_without_f64_mirrors() {
        macro_rules! assert_typed_scalar {
            ($storage:expr, $expected:expr) => {{
                let tensor = Tensor::new_integer($storage, vec![1, 1]).expect("scalar tensor");
                assert_eq!(value_to_f64(&Value::Tensor(tensor)).unwrap(), $expected);
            }};
        }

        assert_typed_scalar!(IntegerStorage::I8(vec![-8]), -8.0);
        assert_typed_scalar!(IntegerStorage::I16(vec![-16]), -16.0);
        assert_typed_scalar!(IntegerStorage::I32(vec![-32]), -32.0);
        assert_typed_scalar!(IntegerStorage::I64(vec![-64]), -64.0);
        assert_typed_scalar!(IntegerStorage::U8(vec![8]), 8.0);
        assert_typed_scalar!(IntegerStorage::U16(vec![16]), 16.0);
        assert_typed_scalar!(IntegerStorage::U32(vec![32]), 32.0);
        assert_typed_scalar!(IntegerStorage::U64(vec![64]), 64.0);
    }

    #[test]
    fn value_to_f64_rejects_wide_integer_values_that_would_round_indices() {
        for value in [
            IntValue::I64(i64::MAX),
            IntValue::U64((1_u64 << 53) + 1),
            IntValue::U64(u64::MAX),
        ] {
            assert!(value_to_f64(&Value::Int(value)).is_err());
        }

        let tensor = Tensor::new_integer(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), vec![1, 1])
            .expect("scalar tensor");
        assert!(value_to_f64(&Value::Tensor(tensor)).is_err());
    }

    #[test]
    fn shared_end_expression_evaluator_uses_base_extent_and_executor_locals() {
        let expression = EndExpr::Sub(Box::new(EndExpr::End), Box::new(EndExpr::Var(2)));
        let value = futures::executor::block_on(resolve_end_expr_value(8, &expression, |local| {
            (local == 2).then_some(Value::Int(IntValue::U8(3)))
        }))
        .unwrap();
        assert_eq!(value, 5.0);
    }
}
