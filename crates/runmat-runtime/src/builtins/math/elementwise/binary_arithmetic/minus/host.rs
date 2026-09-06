use super::*;

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

fn scalar_minus_value(left: &Value, right: &Value) -> Option<Value> {
    if matches!(left, Value::Tensor(_) | Value::ComplexTensor(_))
        || matches!(right, Value::Tensor(_) | Value::ComplexTensor(_))
    {
        return None;
    }
    let left = scalar_complex_value(left).or_else(|| scalar_real_value(left).map(|v| (v, 0.0)))?;
    let right =
        scalar_complex_value(right).or_else(|| scalar_real_value(right).map(|v| (v, 0.0)))?;
    if left.1 != 0.0 || right.1 != 0.0 {
        Some(Value::Complex(left.0 - right.0, left.1 - right.1))
    } else {
        Some(Value::Num(left.0 - right.0))
    }
}

pub(super) fn minus_host(lhs: Value, rhs: Value) -> BuiltinResult<Value> {
    if let Some(result) = symbolic_binary(&lhs, &rhs, SymbolicBinaryOp::Sub) {
        return Ok(result);
    }
    if let Some(result) =
        try_typed_sparse_integer_binary(&lhs, &rhs, SparseBinaryOp::Sub, BUILTIN_NAME)
    {
        return result;
    }
    if let Some(result) = try_sparse_binary(&lhs, &rhs, SparseBinaryOp::Sub, BUILTIN_NAME) {
        return result;
    }
    if (is_real_integer_operand(&lhs) && is_complex_operand(&rhs))
        || (is_complex_operand(&lhs) && is_real_integer_operand(&rhs))
    {
        return Err(builtin_error("complex integer arithmetic is not supported"));
    }
    if let Some(result) = try_integer_binary(&lhs, &rhs, IntegerBinaryOp::Subtract, BUILTIN_NAME)
        .map_err(builtin_error)?
    {
        return Ok(result);
    }
    if let Some(result) = scalar_minus_value(&lhs, &rhs) {
        return Ok(result);
    }
    match (classify_operand(lhs)?, classify_operand(rhs)?) {
        (MinusOperand::Real(a), MinusOperand::Real(b)) => minus_real_real(a, b),
        (MinusOperand::Complex(a), MinusOperand::Complex(b)) => minus_complex_complex(&a, &b),
        (MinusOperand::Complex(a), MinusOperand::Real(b)) => minus_complex_real(&a, &b),
        (MinusOperand::Real(a), MinusOperand::Complex(b)) => minus_real_complex(&a, &b),
    }
}

pub(super) fn is_real_integer_operand(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
}

fn is_complex_operand(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
}

enum MinusOperand {
    Real(Tensor),
    Complex(ComplexTensor),
}

fn classify_operand(value: Value) -> BuiltinResult<MinusOperand> {
    match value {
        Value::Tensor(t) => Ok(MinusOperand::Real(t)),
        Value::Num(n) => Ok(MinusOperand::Real(
            Tensor::new(vec![n], vec![1, 1]).map_err(|e| builtin_error(format!("minus: {e}")))?,
        )),
        Value::Bool(b) => Ok(MinusOperand::Real(
            Tensor::new(vec![if b { 1.0 } else { 0.0 }], vec![1, 1])
                .map_err(|e| builtin_error(format!("minus: {e}")))?,
        )),
        Value::LogicalArray(logical) => Ok(MinusOperand::Real(
            tensor::logical_to_tensor(&logical)
                .map_err(|e| builtin_error(format!("minus: {e}")))?,
        )),
        Value::CharArray(chars) => Ok(MinusOperand::Real(char_array_to_tensor(&chars)?)),
        Value::Complex(re, im) => Ok(MinusOperand::Complex(
            ComplexTensor::new(vec![(re, im)], vec![1, 1])
                .map_err(|e| builtin_error(format!("minus: {e}")))?,
        )),
        Value::ComplexTensor(ct) => Ok(MinusOperand::Complex(ct)),
        Value::GpuTensor(_) => Err(minus_error(&MINUS_ERROR_INTERNAL)),
        other => Err(minus_error_with_detail(
            &MINUS_ERROR_INVALID_INPUT,
            format!(
                "unsupported operand type {:?}; expected numeric or logical data",
                other
            ),
        )),
    }
}

fn minus_real_real(lhs: Tensor, rhs: Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| minus_error_with_detail(&MINUS_ERROR_SIZE_MISMATCH, &err))?;
    let output_shape = plan.output_shape().to_vec();
    let lhs = lhs
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    let rhs = rhs
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    let output = match (lhs, rhs) {
        (NumericStorage::F32(lhs), NumericStorage::F32(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = lhs[lhs_index] - rhs[rhs_index];
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(lhs), NumericStorage::F64(rhs)) => {
            let mut output = vec![0.0f64; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = lhs[lhs_index] - rhs[rhs_index];
            }
            NumericStorage::F64(output)
        }
        (NumericStorage::F32(lhs), NumericStorage::F64(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (f64::from(lhs[lhs_index]) - rhs[rhs_index]) as f32;
            }
            NumericStorage::F32(output)
        }
        (NumericStorage::F64(lhs), NumericStorage::F32(rhs)) => {
            let mut output = vec![0.0f32; plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (lhs[lhs_index] - f64::from(rhs[rhs_index])) as f32;
            }
            NumericStorage::F32(output)
        }
        _ => {
            return Err(builtin_error(
                "minus: integer operands did not use the exact integer arithmetic path",
            ))
        }
    };
    let tensor = Tensor::from_numeric_storage(output, output_shape)
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    Ok(tensor::tensor_into_value(tensor))
}

fn minus_complex_complex(lhs: &ComplexTensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| minus_error_with_detail(&MINUS_ERROR_SIZE_MISMATCH, &err))?;
    let output = match (lhs.complex_storage(), rhs.complex_storage()) {
        (ComplexStorage::F64(lhs), ComplexStorage::F64(rhs)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (
                    lhs[lhs_index].0 - rhs[rhs_index].0,
                    lhs[lhs_index].1 - rhs[rhs_index].1,
                );
            }
            ComplexStorage::F64(output.into())
        }
        (ComplexStorage::F32(lhs), ComplexStorage::F32(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (
                    lhs[lhs_index].0 - rhs[rhs_index].0,
                    lhs[lhs_index].1 - rhs[rhs_index].1,
                );
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F32(lhs), ComplexStorage::F64(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (
                    (f64::from(lhs[lhs_index].0) - rhs[rhs_index].0) as f32,
                    (f64::from(lhs[lhs_index].1) - rhs[rhs_index].1) as f32,
                );
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F64(lhs), ComplexStorage::F32(rhs)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                output[output_index] = (
                    (lhs[lhs_index].0 - f64::from(rhs[rhs_index].0)) as f32,
                    (lhs[lhs_index].1 - f64::from(rhs[rhs_index].1)) as f32,
                );
            }
            ComplexStorage::F32(output.into())
        }
        _ => {
            return Err(builtin_error(
                "minus: complex integer arithmetic is not supported",
            ))
        }
    };
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn minus_complex_real(lhs: &ComplexTensor, rhs: &Tensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| minus_error_with_detail(&MINUS_ERROR_SIZE_MISMATCH, &err))?;
    let rhs = rhs
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    let output = subtract_complex_real_storage(lhs.complex_storage(), &rhs, &plan, true)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn minus_real_complex(lhs: &Tensor, rhs: &ComplexTensor) -> BuiltinResult<Value> {
    let plan = BroadcastPlan::new(&lhs.shape, &rhs.shape)
        .map_err(|err| minus_error_with_detail(&MINUS_ERROR_SIZE_MISMATCH, &err))?;
    let lhs = lhs
        .clone()
        .into_numeric_storage()
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    let output = subtract_complex_real_storage(rhs.complex_storage(), &lhs, &plan, false)?;
    let tensor = ComplexTensor::from_complex_storage(output, plan.output_shape().to_vec())
        .map_err(|e| builtin_error(format!("minus: {e}")))?;
    Ok(complex_tensor_into_value(tensor))
}

fn subtract_complex_real_storage(
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
    let subtract_f64 = |complex: (f64, f64), real: f64| {
        if complex_is_left {
            (complex.0 - real, complex.1)
        } else {
            (real - complex.0, -complex.1)
        }
    };
    let subtract_f32 = |complex: (f32, f32), real: f32| {
        if complex_is_left {
            (complex.0 - real, complex.1)
        } else {
            (real - complex.0, -complex.1)
        }
    };
    Ok(match (complex, real) {
        (ComplexStorage::F64(complex), NumericStorage::F64(real)) => {
            let mut output = vec![(0.0f64, 0.0f64); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                output[output_index] =
                    subtract_f64(complex[complex_index].into(), real[real_index]);
            }
            ComplexStorage::F64(output.into())
        }
        (ComplexStorage::F32(complex), NumericStorage::F32(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                output[output_index] =
                    subtract_f32(complex[complex_index].into(), real[real_index]);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F32(complex), NumericStorage::F64(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                let value = complex[complex_index];
                let result =
                    subtract_f64((f64::from(value.0), f64::from(value.1)), real[real_index]);
                output[output_index] = (result.0 as f32, result.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        (ComplexStorage::F64(complex), NumericStorage::F32(real)) => {
            let mut output = vec![(0.0f32, 0.0f32); plan.len()];
            for (output_index, lhs_index, rhs_index) in plan.iter() {
                let (complex_index, real_index) = indices(lhs_index, rhs_index);
                let result =
                    subtract_f64(complex[complex_index].into(), f64::from(real[real_index]));
                output[output_index] = (result.0 as f32, result.1 as f32);
            }
            ComplexStorage::F32(output.into())
        }
        _ => {
            return Err(builtin_error(
                "minus: integer operands did not use the exact integer arithmetic path",
            ))
        }
    })
}

fn char_array_to_tensor(chars: &CharArray) -> BuiltinResult<Tensor> {
    let data: Vec<f64> = chars.data.iter().map(|&ch| ch as u32 as f64).collect();
    Tensor::new(data, vec![chars.rows, chars.cols])
        .map_err(|e| builtin_error(format!("minus: {e}")))
}
