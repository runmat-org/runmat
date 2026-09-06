use super::*;

fn scalar_real_value(value: &Value) -> Option<f64> {
    match value {
        Value::Num(value) => Some(*value),
        Value::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        Value::Tensor(value) if tensor::is_scalar_tensor(value) => {
            Some(tensor::tensor_value_f64(value, 0))
        }
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
        Value::ComplexTensor(value) if tensor::is_scalar_complex_tensor(value) => {
            let scalar = tensor::complex_tensor_value_complex64(value, 0);
            Some((scalar.re, scalar.im))
        }
        _ => None,
    }
}

fn scalar_times_value(left: &Value, right: &Value) -> Option<Value> {
    if matches!(left, Value::Tensor(_) | Value::ComplexTensor(_))
        || matches!(right, Value::Tensor(_) | Value::ComplexTensor(_))
    {
        return None;
    }
    let left = scalar_complex_value(left).or_else(|| scalar_real_value(left).map(|v| (v, 0.0)))?;
    let right =
        scalar_complex_value(right).or_else(|| scalar_real_value(right).map(|v| (v, 0.0)))?;
    if left.1 != 0.0 || right.1 != 0.0 {
        Some(Value::Complex(
            left.0 * right.0 - left.1 * right.1,
            left.0 * right.1 + left.1 * right.0,
        ))
    } else {
        Some(Value::Num(left.0 * right.0))
    }
}

pub(crate) fn times_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if let Some(result) = symbolic_binary(&lhs, &rhs, SymbolicBinaryOp::Mul) {
        return Ok(result);
    }
    if let Some(result) =
        try_typed_sparse_integer_binary(&lhs, &rhs, SparseBinaryOp::Mul, BUILTIN_NAME)
    {
        return result;
    }
    if let Some(result) = try_sparse_binary(&lhs, &rhs, SparseBinaryOp::Mul, BUILTIN_NAME) {
        return result;
    }
    if let Some(result) = try_integer_binary(&lhs, &rhs, IntegerBinaryOp::Multiply, BUILTIN_NAME)
        .map_err(builtin_error)?
    {
        return Ok(result);
    }
    if let Some(result) = scalar_times_value(&lhs, &rhs) {
        return Ok(result);
    }
    match (classify_operand(lhs)?, classify_operand(rhs)?) {
        (TimesOperand::Real(a), TimesOperand::Real(b)) => times_real_real(a, b),
        (TimesOperand::Complex(a), TimesOperand::Complex(b)) => times_complex_complex(&a, &b),
        (TimesOperand::Complex(a), TimesOperand::Real(b)) => times_complex_real(&a, &b),
        (TimesOperand::Real(a), TimesOperand::Complex(b)) => times_real_complex(&a, &b),
    }
}

fn times_real_real(lhs: Tensor, rhs: Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| times_error_with_detail(&TIMES_ERROR_SIZE_MISMATCH, &err))?;
    let output_shape = plan.output_shape().to_vec();
    let lhs = lhs
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    let rhs = rhs
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    let output = match (lhs, rhs) {
        (NumericStorage::F32(lhs), NumericStorage::F32(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = lhs[lhs_index] * rhs[rhs_index];
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(lhs), NumericStorage::F64(rhs)) => {
            let mut output = vec![0.0f64; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = lhs[lhs_index] * rhs[rhs_index];
            }
            NumericStorage::F64(output)
        }
        (NumericStorage::F32(lhs), NumericStorage::F64(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (f64::from(lhs[lhs_index]) * rhs[rhs_index]) as f32;
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(lhs), NumericStorage::F32(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (lhs[lhs_index] * f64::from(rhs[rhs_index])) as f32;
            }
            NumericStorage::F32(output)
        }
        _ => {
            return Err(builtin_error(
                "times: integer operands did not use the exact integer arithmetic path",
            ))
        }
    };
    let tensor = Tensor::from_numeric_storage(output, output_shape)
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn times_complex_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| times_error_with_detail(&TIMES_ERROR_SIZE_MISMATCH, &err))?;
    let output = match (lhs.complex_storage(), rhs.complex_storage()) {
        (ComplexStorage::F64(lhs), ComplexStorage::F64(rhs)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = multiply_complex_f64(lhs[lhs_index], rhs[rhs_index]);
            }
            ComplexStorage::F64(output.into())
        }
        (ComplexStorage::F32(lhs), ComplexStorage::F32(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = multiply_complex_f32(lhs[lhs_index], rhs[rhs_index]);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F32(lhs), ComplexStorage::F64(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let lhs = (f64::from(lhs[lhs_index].0), f64::from(lhs[lhs_index].1));
                let value = multiply_complex_f64(lhs, rhs[rhs_index]);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F64(lhs), ComplexStorage::F32(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let rhs = (f64::from(rhs[rhs_index].0), f64::from(rhs[rhs_index].1));
                let value = multiply_complex_f64(lhs[lhs_index], rhs);
                output[output_index] = (value.0 as f32, value.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        _ => {
            return Err(builtin_error(
                "times: complex integer arithmetic is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn times_complex_real(lhs: &ComplexTensor, rhs: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| times_error_with_detail(&TIMES_ERROR_SIZE_MISMATCH, &err))?;
    let rhs = rhs
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    let output = multiply_complex_real_storage(lhs.complex_storage(), &rhs, &plan, true)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn times_real_complex(lhs: &Tensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| times_error_with_detail(&TIMES_ERROR_SIZE_MISMATCH, &err))?;
    let lhs = lhs
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    let output = multiply_complex_real_storage(rhs.complex_storage(), &lhs, &plan, false)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("times: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn multiply_complex_real_storage(
    complex: &ComplexStorage,
    real: &NumericStorage,
    plan: &BroadcastPlan,
    complex_is_left: bool,
) -> BuiltinResult<ComplexStorage> {
    let indices = |lhs_index: usize, rhs_index: usize| {
        if complex_is_left {
            (lhs_index, rhs_index)
        } else {
            (rhs_index, lhs_index)
        }
    };
    Ok(match (complex, real) {
        (ComplexStorage::F64(complex), NumericStorage::F64(real)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (value.0 * scalar, value.1 * scalar);
            }
            ComplexStorage::F64(output.into())
        }
        (ComplexStorage::F32(complex), NumericStorage::F32(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (value.0 * scalar, value.1 * scalar);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F32(complex), NumericStorage::F64(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                let value = complex[complex_index];
                let scalar = real[real_index];
                output[output_index] = (
                    (f64::from(value.0) * scalar) as f32,
                    (f64::from(value.1) * scalar) as f32,
                );
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F64(complex), NumericStorage::F32(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                let value = complex[complex_index];
                let scalar = f64::from(real[real_index]);
                output[output_index] = ((value.0 * scalar) as f32, (value.1 * scalar) as f32);
            }
            ComplexStorage::F32(output.into())
        }
        _ => {
            return Err(builtin_error(
                "times: integer operands did not use the exact integer arithmetic path",
            ))
        }
    })
}

fn multiply_complex_f64(lhs: impl Into<(f64, f64)>, rhs: impl Into<(f64, f64)>) -> (f64, f64) {
    let lhs = lhs.into();
    let rhs = rhs.into();
    (lhs.0 * rhs.0 - lhs.1 * rhs.1, lhs.0 * rhs.1 + lhs.1 * rhs.0)
}

fn multiply_complex_f32(lhs: impl Into<(f32, f32)>, rhs: impl Into<(f32, f32)>) -> (f32, f32) {
    let lhs = lhs.into();
    let rhs = rhs.into();
    (lhs.0 * rhs.0 - lhs.1 * rhs.1, lhs.0 * rhs.1 + lhs.1 * rhs.0)
}

enum TimesOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> BuiltinResult<TimesOperand> {
    match value {
        Value::Tensor(t) => Ok(TimesOperand::Real(t)),
        Value::Num(n) => Ok(TimesOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| builtin_error(format!("times: {e}")))?,
        )),
        Value::Bool(b) => Ok(TimesOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| builtin_error(format!("times: {e}")))?,
        )),
        Value::LogicalArray(logical) => Ok(TimesOperand::Real(
            tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("times: {e}")))?,
        )),
        Value::CharArray(chars) => Ok(TimesOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(TimesOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("times: {e}")))?,
        )),
        Value::ComplexTensor(ct) => Ok(TimesOperand::Complex(ct)),
        Value::GpuTensor(_) => Err(times_error(&TIMES_ERROR_INTERNAL)),
        other => Err(times_error_with_detail(
            &TIMES_ERROR_INVALID_INPUT,
            format!(
                "unsupported operand type {:?}; expected numeric or logical data",
                other
            ),
        )),
    }
}

fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| builtin_error(format!("times: {e}")))
}
