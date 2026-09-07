use num_complex::Complex64;
use runmat_value::{CharArray, ComplexTensor, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::math::elementwise::integer_arithmetic::{try_integer_binary, IntegerBinaryOp};
use crate::builtins::math::symbolic::{symbolic_binary, SymbolicBinaryOp};
use crate::BuiltinResult;

use super::DivisionContext;

pub(super) fn execute(context: DivisionContext, lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if let Some(result) = symbolic_binary(&lhs, &rhs, SymbolicBinaryOp::Div) {
        return Ok(result);
    }
    if let Some(result) =
        try_integer_binary(&lhs, &rhs, IntegerBinaryOp::Divide, context.identity.name)
            .map_err(|detail| context.internal_error(detail))?
    {
        return Ok(result);
    }
    if let Some(result) = scalar_divide_value(&lhs, &rhs) {
        return Ok(result);
    }
    match (
        classify_operand(context, lhs)?,
        classify_operand(context, rhs)?,
    ) {
        (DivisionOperand::Real(a), DivisionOperand::Real(b)) => {
            real::division_real_real(context, a, b)
        }
        (DivisionOperand::Complex(a), DivisionOperand::Complex(b)) => {
            complex::division_complex_complex(context, &a, &b)
        }
        (DivisionOperand::Complex(a), DivisionOperand::Real(b)) => {
            complex::division_complex_real(context, &a, &b)
        }
        (DivisionOperand::Real(a), DivisionOperand::Complex(b)) => {
            complex::division_real_complex(context, &a, &b)
        }
    }
}

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::LogicalArray(value) if value.data.len() == 1 => {
            Some(if value.data[0] != 0 { 1.0 } else { 0.0 })
        }
        Value::CharArray(value) if value.rows * value.cols == 1 => Some(
            value
                .data
                .first()
                .map(|&character| character as u32 as f64)
                .unwrap_or(0.0),
        ),
        _ => None,
    }
}

fn scalar_complex_value(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Complex(real, imaginary) => Some((*real, *imaginary)),
        _ => None,
    }
}

fn scalar_divide_value(numerator: &Value, denominator: &Value) -> Option<Value> {
    if matches!(numerator, Value::Tensor(_) | Value::ComplexTensor(_))
        || matches!(denominator, Value::Tensor(_) | Value::ComplexTensor(_))
    {
        return None;
    }
    let numerator = scalar_complex_value(numerator)
        .or_else(|| scalar_real_value(numerator).map(|value| (value, 0.0)))?;
    let denominator = scalar_complex_value(denominator)
        .or_else(|| scalar_real_value(denominator).map(|value| (value, 0.0)))?;
    if numerator.1 != 0.0 || denominator.1 != 0.0 {
        let quotient =
            Complex64::new(numerator.0, numerator.1) / Complex64::new(denominator.0, denominator.1);
        return Some(Value::Complex(quotient.re, quotient.im));
    }
    Some(Value::Num(numerator.0 / denominator.0))
}

mod complex;
mod real;

enum DivisionOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(context: DivisionContext, value: Value) -> BuiltinResult<DivisionOperand> {
    match value {
        Value::Tensor(t) => Ok(DivisionOperand::Real(t)),
        Value::Num(n) => Ok(DivisionOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|error| context.internal_error(error))?,
        )),
        Value::Bool(b) => Ok(DivisionOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|error| context.internal_error(error))?,
        )),
        Value::LogicalArray(logical) => Ok(DivisionOperand::Real(
            tensor::logical_to_tensor(&logical).map_err(|error| context.internal_error(error))?,
        )),
        Value::CharArray(chars) => Ok(DivisionOperand::Real(char_array_to_tensor(
            context, &chars,
        )?)),
        Value::Complex(re, im) => Ok(DivisionOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|error| context.internal_error(error))?,
        )),
        Value::ComplexTensor(ct) => Ok(DivisionOperand::Complex(ct)),
        Value::GpuTensor(_) => Err(context.error(context.internal)),
        other => Err(context.error_with_detail(
            context.invalid_input,
            format!(
                "unsupported operand type {:?}; expected numeric or logical data",
                other
            ),
        )),
    }
}

fn char_array_to_tensor(context: DivisionContext, chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols]).map_err(|error| context.internal_error(error))
}
