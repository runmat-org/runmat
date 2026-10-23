use runmat_builtins::{POWER_ERROR_INTERNAL, POWER_ERROR_INVALID_INPUT, POWER_ERROR_SIZE_MISMATCH};
use runmat_value::{CharArray, ComplexTensor, Tensor, Value};

use crate::builtins::common::tensor;
use crate::builtins::math::elementwise::integer_arithmetic::{try_integer_binary, IntegerBinaryOp};
use crate::builtins::math::symbolic::{
    symbolic_binary, symbolic_binary_broadcast, SymbolicBinaryOp,
};
use crate::BuiltinResult;

use super::{
    builtin_error, complex_pow_scalar, power_error, power_error_with_detail, BUILTIN_NAME,
};

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(n) => Some(*n),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::LogicalArray(l) if l.data.len() == 1 => Some(if l.data[0] != 0 { 1.0 } else { 0.0 }),
        Value::CharArray(ca) if ca.data().len() == 1 => Some(
            ca.data()
                .first()
                .map(|&ch| f64::from(ch.raw()))
                .unwrap_or(0.0),
        ),
        _ => None,
    }
}

fn scalar_complex_value(value: &Value) -> Option<(f64, f64)> {
    match value {
        Value::Complex(re, im) => Some((*re, *im)),
        _ => None,
    }
}

fn scalar_power_value(lhs: &Value, rhs: &Value) -> Option<Value> {
    if matches!(lhs, Value::Tensor(_) | Value::ComplexTensor(_))
        || matches!(rhs, Value::Tensor(_) | Value::ComplexTensor(_))
    {
        return None;
    }
    let base_is_complex = matches!(lhs, Value::Complex(_, _) | Value::ComplexTensor(_));
    let exp_is_complex = matches!(rhs, Value::Complex(_, _) | Value::ComplexTensor(_));
    let base = scalar_complex_value(lhs).or_else(|| scalar_real_value(lhs).map(|v| (v, 0.0)))?;
    let exp = scalar_complex_value(rhs).or_else(|| scalar_real_value(rhs).map(|v| (v, 0.0)))?;
    let (br, bi) = base;
    let (er, ei) = exp;
    if base_is_complex || exp_is_complex || bi != 0.0 || ei != 0.0 {
        let (re, im) = complex_pow_scalar(br, bi, er, ei);
        return Some(Value::Complex(re, im));
    }
    let (domain, (re, im)) = super::domain::evaluate_f64(br, er);
    if domain == super::domain::PowerResultDomain::Complex {
        Some(Value::Complex(re, im))
    } else {
        Some(Value::Num(re))
    }
}

pub(super) fn power_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if let Some(result) = symbolic_binary_broadcast(&lhs, &rhs, SymbolicBinaryOp::Pow)
        .map_err(|err| power_error_with_detail(&POWER_ERROR_SIZE_MISMATCH, &err))?
    {
        return Ok(result);
    }
    if let Some(result) = symbolic_binary(&lhs, &rhs, SymbolicBinaryOp::Pow) {
        return Ok(result);
    }
    // Character arrays participate in numeric power through their Unicode code
    // points. An exact-integer operand on the other side must not claim this
    // operation before the character operand is converted below.
    if !matches!(lhs, Value::CharArray(_)) && !matches!(rhs, Value::CharArray(_)) {
        if let Some(result) =
            try_integer_binary(&lhs, &rhs, IntegerBinaryOp::Power, BUILTIN_NAME)
                .map_err(|error| power_error_with_detail(&POWER_ERROR_INVALID_INPUT, error))?
        {
            return Ok(result);
        }
    }
    if let Some(result) = scalar_power_value(&lhs, &rhs) {
        return Ok(result);
    }
    match (classify_operand(lhs)?, classify_operand(rhs)?) {
        (PowerOperand::Real(a), PowerOperand::Real(b)) => real::power_real_real(a, b),
        (PowerOperand::Complex(a), PowerOperand::Complex(b)) => {
            complex::power_complex_complex(&a, &b)
        }
        (PowerOperand::Complex(a), PowerOperand::Real(b)) => complex::power_complex_real(&a, &b),
        (PowerOperand::Real(a), PowerOperand::Complex(b)) => complex::power_real_complex(&a, &b),
    }
}

mod complex;
mod real;

enum PowerOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> BuiltinResult<PowerOperand> {
    match value {
        Value::Tensor(t) => Ok(PowerOperand::Real(t)),
        Value::Num(n) => Ok(PowerOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| builtin_error(format!("power: {e}")))?,
        )),
        Value::Bool(b) => Ok(PowerOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| builtin_error(format!("power: {e}")))?,
        )),
        Value::LogicalArray(logical) => Ok(PowerOperand::Real(
            tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("power: {e}")))?,
        )),
        Value::CharArray(chars) => Ok(PowerOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(PowerOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("power: {e}")))?,
        )),
        Value::ComplexTensor(ct) => Ok(PowerOperand::Complex(ct)),
        Value::GpuTensor(_) => Err(power_error(&POWER_ERROR_INTERNAL)),
        other => Err(power_error_with_detail(
            &POWER_ERROR_INVALID_INPUT,
            format!(
                "unsupported operand type {:?}; expected numeric, logical, or char data",
                other
            ),
        )),
    }
}

pub(super) fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars
        .to_column_major()
        .iter()
        .map(|&ch| f64::from(ch.raw()))
        .collect();
    Tensor::new(data, chars.shape.clone()).map_err(|e| builtin_error(format!("power: {e}")))
}
