use num_bigint::BigInt;
use runmat_value::{IntValue, NumericScalar};
pub(crate) mod digits;
pub(crate) mod int;
pub(crate) mod limit;
pub(crate) mod piecewise;
pub(crate) mod sym;
pub(crate) mod syms;
pub(crate) mod vpa;

use runmat_types::symbolic::is_valid_symbolic_identifier;
use runmat_value::SymbolicFunction;
use runmat_value::{SymbolicArray, SymbolicExpr, Tensor, Value};

use crate::builtins::common::broadcast::BroadcastPlan;
use crate::builtins::common::shape::is_scalar_shape;
use crate::builtins::common::tensor as tensor_utils;

#[derive(Debug, Clone, Copy)]
pub(crate) enum SymbolicBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Eq,
}

pub(crate) fn symbolic_named_binary(lhs: &Value, rhs: &Value, name: &str) -> Option<Value> {
    let (lhs, rhs) = symbolic_binary_operands(lhs, rhs)?;
    Some(symbolic_expr_to_value(SymbolicExpr::function_call(
        name,
        vec![lhs, rhs],
    )))
}

pub(crate) fn symbolic_binary(lhs: &Value, rhs: &Value, op: SymbolicBinaryOp) -> Option<Value> {
    let (lhs, rhs) = symbolic_binary_operands(lhs, rhs)?;
    let expr = match op {
        SymbolicBinaryOp::Add => SymbolicExpr::add_expr(lhs, rhs),
        SymbolicBinaryOp::Sub => SymbolicExpr::sub_expr(lhs, rhs),
        SymbolicBinaryOp::Mul => SymbolicExpr::mul_expr(lhs, rhs),
        SymbolicBinaryOp::Div => SymbolicExpr::div_expr(lhs, rhs),
        SymbolicBinaryOp::Pow => SymbolicExpr::pow_expr(lhs, rhs),
        SymbolicBinaryOp::Eq => SymbolicExpr::equation(lhs, rhs),
    };
    Some(symbolic_expr_to_value(expr))
}

pub(crate) fn symbolic_binary_broadcast(
    lhs: &Value,
    rhs: &Value,
    op: SymbolicBinaryOp,
) -> Result<Option<Value>, String> {
    if !contains_symbolic_value(lhs) && !contains_symbolic_value(rhs) {
        return Ok(None);
    }

    let Some(lhs) = SymbolicOperand::from_value(lhs) else {
        return Ok(None);
    };
    let Some(rhs) = SymbolicOperand::from_value(rhs) else {
        return Ok(None);
    };

    let plan = BroadcastPlan::new(lhs.shape(), rhs.shape())?;
    let mut data = Vec::with_capacity(plan.len());
    for (_, lhs_idx, rhs_idx) in plan.iter() {
        let lhs_expr = lhs.expr_at(lhs_idx).clone();
        let rhs_expr = rhs.expr_at(rhs_idx).clone();
        data.push(match op {
            SymbolicBinaryOp::Add => SymbolicExpr::add_expr(lhs_expr, rhs_expr),
            SymbolicBinaryOp::Sub => SymbolicExpr::sub_expr(lhs_expr, rhs_expr),
            SymbolicBinaryOp::Mul => SymbolicExpr::mul_expr(lhs_expr, rhs_expr),
            SymbolicBinaryOp::Div => SymbolicExpr::div_expr(lhs_expr, rhs_expr),
            SymbolicBinaryOp::Pow => SymbolicExpr::pow_expr(lhs_expr, rhs_expr),
            SymbolicBinaryOp::Eq => SymbolicExpr::equation(lhs_expr, rhs_expr),
        });
    }

    if data.len() == 1 && is_scalar_shape(plan.output_shape()) {
        return Ok(Some(Value::Symbolic(data.remove(0))));
    }

    SymbolicArray::new(data, plan.output_shape().to_vec())
        .map(Value::SymbolicArray)
        .map(Some)
}

fn contains_symbolic_value(value: &Value) -> bool {
    matches!(value, Value::Symbolic(_) | Value::SymbolicArray(_))
}

struct SymbolicOperand {
    data: Vec<SymbolicExpr>,
    shape: Vec<usize>,
}

impl SymbolicOperand {
    fn scalar(expr: SymbolicExpr) -> Self {
        Self {
            data: vec![expr],
            shape: vec![1, 1],
        }
    }

    fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Symbolic(expr) => Some(Self::scalar(expr.clone())),
            Value::SymbolicArray(array) => Some(Self {
                data: array.data.clone(),
                shape: normalize_symbolic_shape(&array.shape),
            }),
            Value::Num(value) => Some(Self::scalar(SymbolicExpr::constant(*value))),
            Value::Int(value) => Some(Self::scalar(SymbolicExpr::constant(value.to_f64()))),
            Value::Bool(value) => Some(Self::scalar(SymbolicExpr::constant(if *value {
                1.0
            } else {
                0.0
            }))),
            Value::Tensor(tensor) => Some(Self {
                data: tensor
                    .materialize_f64()
                    .iter()
                    .copied()
                    .map(SymbolicExpr::constant)
                    .collect(),
                shape: normalize_symbolic_shape(&tensor.shape),
            }),
            Value::LogicalArray(array) => Some(Self {
                data: array
                    .data
                    .iter()
                    .map(|value| SymbolicExpr::constant(if *value == 0 { 0.0 } else { 1.0 }))
                    .collect(),
                shape: normalize_symbolic_shape(&array.shape),
            }),
            _ => None,
        }
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn expr_at(&self, index: usize) -> &SymbolicExpr {
        &self.data[index]
    }
}

fn normalize_symbolic_shape(shape: &[usize]) -> Vec<usize> {
    if shape.len() == 1 && shape[0] != 1 {
        vec![1, shape[0]]
    } else if is_scalar_shape(shape) {
        vec![1, 1]
    } else {
        shape.to_vec()
    }
}

fn symbolic_binary_operands(lhs: &Value, rhs: &Value) -> Option<(SymbolicExpr, SymbolicExpr)> {
    if !matches!(lhs, Value::Symbolic(_)) && !matches!(rhs, Value::Symbolic(_)) {
        return None;
    }
    let lhs = value_to_symbolic_scalar(lhs)?;
    let rhs = value_to_symbolic_scalar(rhs)?;
    Some((lhs, rhs))
}

pub(crate) fn symbolic_function(value: &Value, function: SymbolicFunction) -> Option<Value> {
    let expr = match value {
        Value::Symbolic(expr) => expr.clone(),
        _ => return None,
    };
    Some(symbolic_expr_to_value(SymbolicExpr::function(
        function, expr,
    )))
}

pub(crate) fn value_to_symbolic_scalar(value: &Value) -> Option<SymbolicExpr> {
    match value {
        Value::Symbolic(expr) => Some(expr.clone()),
        Value::Num(value) => Some(SymbolicExpr::constant(*value)),
        Value::Int(value) => symbolic_integer(value),
        Value::Bool(value) => Some(SymbolicExpr::constant(if *value { 1.0 } else { 0.0 })),
        Value::Tensor(tensor) if tensor_utils::is_scalar_tensor(tensor) => {
            match tensor.numeric_value_at(0)? {
                NumericScalar::F64(value) => Some(SymbolicExpr::constant(value)),
                NumericScalar::F32(value) => Some(SymbolicExpr::constant(value as f64)),
                value => value
                    .into_int_value()
                    .and_then(|value| symbolic_integer(&value)),
            }
        }
        _ => None,
    }
}

fn symbolic_integer(value: &IntValue) -> Option<SymbolicExpr> {
    let numerator = match value {
        IntValue::I8(value) => BigInt::from(*value),
        IntValue::I16(value) => BigInt::from(*value),
        IntValue::I32(value) => BigInt::from(*value),
        IntValue::I64(value) => BigInt::from(*value),
        IntValue::U8(value) => BigInt::from(*value),
        IntValue::U16(value) => BigInt::from(*value),
        IntValue::U32(value) => BigInt::from(*value),
        IntValue::U64(value) => BigInt::from(*value),
    };
    SymbolicExpr::rational(numerator, BigInt::from(1_u8))
}

pub(crate) fn symbolic_expr_to_value(expr: SymbolicExpr) -> Value {
    Value::Symbolic(expr)
}

pub(crate) fn symbolic_variable_name_from_value(value: &Value) -> Option<String> {
    match value {
        Value::Symbolic(expr) => expr.variable_name().map(ToOwned::to_owned),
        _ => text_scalar(value).map(|text| text.trim().to_string()),
    }
    .filter(|name| is_valid_symbolic_identifier(name))
}

pub(crate) fn empty_return_value() -> Value {
    Value::Tensor(Tensor::zeros(vec![0, 0]))
}

pub(crate) fn text_scalar(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::CharArray(chars) if chars.rows == 1 => Some(chars.data.iter().collect()),
        Value::StringArray(array) if array.data.len() == 1 => Some(array.data[0].clone()),
        _ => None,
    }
}

pub(crate) fn is_valid_identifier(name: &str) -> bool {
    is_valid_symbolic_identifier(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use runmat_value::IntegerStorage;

    #[test]
    fn symbolic_scalar_reads_typed_integer_tensor_storage_exactly() {
        let tensor =
            Tensor::new_integer(IntegerStorage::U16(vec![257]), vec![1, 1]).expect("tensor");

        let expr =
            value_to_symbolic_scalar(&Value::Tensor(tensor)).expect("symbolic scalar conversion");
        assert_eq!(expr.constant_value(), Some(257.0));
    }

    #[test]
    fn symbolic_scalars_preserve_wide_integer_values() {
        let scalar = value_to_symbolic_scalar(&Value::Int(IntValue::U64(u64::MAX)))
            .expect("symbolic integer scalar");
        assert_eq!(scalar.to_string(), u64::MAX.to_string());

        let tensor = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN]), vec![1, 1])
            .expect("integer tensor");
        let tensor_value =
            value_to_symbolic_scalar(&Value::Tensor(tensor)).expect("symbolic integer tensor");
        assert_eq!(tensor_value.to_string(), i64::MIN.to_string());
    }
}
