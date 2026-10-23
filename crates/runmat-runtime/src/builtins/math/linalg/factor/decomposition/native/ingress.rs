//! Selected decomposition ingress owner.
use super::*;
pub(in crate::builtins::math::linalg::factor::decomposition) fn exact_integer(
    value: IntValue,
) -> i128 {
    match value {
        IntValue::I8(v) => i128::from(v),
        IntValue::I16(v) => i128::from(v),
        IntValue::I32(v) => i128::from(v),
        IntValue::I64(v) => i128::from(v),
        IntValue::U8(v) => i128::from(v),
        IntValue::U16(v) => i128::from(v),
        IntValue::U32(v) => i128::from(v),
        IntValue::U64(v) => i128::from(v),
    }
}
pub(in crate::builtins::math::linalg::factor::decomposition) fn checked_double(
    value: NumericScalar,
) -> BuiltinResult<f64> {
    checked_double_for(value, None)
}
fn checked_double_for(value: NumericScalar, kind: Option<DecompositionType>) -> BuiltinResult<f64> {
    match value {
        NumericScalar::F32(v) => Ok(f64::from(v)),
        NumericScalar::F64(v) => Ok(v),
        other => {
            let value = other.into_int_value().expect("integer scalar");
            value.to_exact_f64().ok_or_else(||invalid(if kind==Some(DecompositionType::Chol) {"decomposition: type 'chol' requires a positive definite coefficient matrix with exactly representable double components"} else {"decomposition: integer coefficient must be exactly representable as double"}))
        }
    }
}
pub(in crate::builtins::math::linalg::factor::decomposition) fn coefficient(
    value: &Value,
    index: usize,
) -> BuiltinResult<(NumericScalar, NumericScalar)> {
    let zero = NumericScalar::F64(0.0);
    match value {
        Value::Tensor(t) => Ok((
            t.numeric_value_at(index)
                .ok_or_else(|| internal("coefficient index out of bounds"))?,
            zero,
        )),
        Value::ComplexTensor(t) => t
            .numeric_value_at(index)
            .ok_or_else(|| internal("complex coefficient index out of bounds")),
        Value::LogicalArray(t) => Ok((
            NumericScalar::U8(
                *t.data
                    .get(index)
                    .ok_or_else(|| internal("logical coefficient index out of bounds"))?,
            ),
            zero,
        )),
        Value::SparseTensor(t) => {
            if let Some(v) = t.complex_value_at(index) {
                Ok((NumericScalar::F64(v.0), NumericScalar::F64(v.1)))
            } else {
                Ok((
                    t.numeric_value_at(index)
                        .ok_or_else(|| internal("sparse coefficient index out of bounds"))?,
                    zero,
                ))
            }
        }
        _ => Err(invalid("expected normalized numeric matrix")),
    }
}
pub(in crate::builtins::math::linalg::factor::decomposition) fn normalize_selected(
    value: Value,
    runtime: &RuntimeContext,
) -> BuiltinResult<(Value, bool)> {
    use runmat_value::{ExactHostComplexTensorAllocation, ExactHostLogicalTensorAllocation};
    match &value {
        Value::Tensor(t) => {
            validate_matrix_shape(t.shape())?;
            return Ok((value, false));
        }
        Value::ComplexTensor(t) => {
            validate_matrix_shape(t.shape())?;
            return Ok((value, false));
        }
        Value::LogicalArray(t) => {
            validate_matrix_shape(&t.shape)?;
            return Ok((value, false));
        }
        Value::SparseTensor(_) => return Ok((value, true)),
        _ => {}
    }
    let bytes = match &value {
        Value::Num(_) => HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[1, 1]),
        Value::Int(v) => HostAllocationPlan::retained_tensor_bytes(v.numeric_dtype(), &[1, 1]),
        Value::Bool(_) => HostAllocationPlan::retained_logical_tensor_bytes(&[1, 1]),
        Value::Complex(_, _) => {
            HostAllocationPlan::retained_complex_tensor_bytes(NumericDType::F64, &[1, 1])
        }
        _ => return Err(invalid("decomposition coefficient matrix must be numeric")),
    }
    .map_err(|e| allocation(e, runtime))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, bytes).map_err(|e| allocation(e, runtime))?,
    )
    .map_err(|e| allocation(e, runtime))?;
    let Some(r) = receipt.as_ref() else {
        return normalize_matrix(value);
    };
    let result = match value {
        Value::Num(v) => {
            let mut a = r
                .exact_output_tensor(runtime, NumericDType::F64, &[1, 1])
                .map_err(|e| allocation(e, runtime))?;
            a.as_f64_slice_mut().expect("double scalar")[0] = v;
            Value::Tensor(
                a.into_tensor()
                    .map_err(|e| allocation(HostAllocationError::Numeric(e), runtime))?,
            )
        }
        Value::Int(v) => {
            let mut a = r
                .exact_output_tensor(runtime, v.numeric_dtype(), &[1, 1])
                .map_err(|e| allocation(e, runtime))?;
            a.set_value(0, NumericScalar::from(v)).map_err(internal)?;
            Value::Tensor(
                a.into_tensor()
                    .map_err(|e| allocation(HostAllocationError::Numeric(e), runtime))?,
            )
        }
        Value::Bool(v) => {
            let mut a = r
                .with_output_allocation(runtime, |p| {
                    ExactHostLogicalTensorAllocation::new(&[1, 1], p)
                })
                .map_err(|e| allocation(e, runtime))?;
            a.as_mut_slice()[0] = u8::from(v);
            Value::LogicalArray(
                a.into_array()
                    .map_err(|e| allocation(HostAllocationError::Numeric(e), runtime))?,
            )
        }
        Value::Complex(re, im) => {
            let mut a = r
                .with_output_allocation(runtime, |p| {
                    ExactHostComplexTensorAllocation::new(NumericDType::F64, &[1, 1], p)
                })
                .map_err(|e| allocation(e, runtime))?;
            a.set(0, re, im)
                .map_err(|e| allocation(HostAllocationError::Numeric(e), runtime))?;
            Value::ComplexTensor(
                a.into_tensor()
                    .map_err(|e| allocation(HostAllocationError::Numeric(e), runtime))?,
            )
        }
        _ => unreachable!("selected scalar kind"),
    };
    r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
    Ok((result, false))
}
pub(in crate::builtins::math::linalg::factor::decomposition) async fn prepare_factor_matrix(
    matrix: &Value,
    kind: DecompositionType,
    triangle: Option<TriangularFlag>,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    let class = super::persistence::expected_factor_class(matrix)?;
    let already_native = match (matrix, class) {
        (Value::Tensor(t), runmat_value::ExactHostSparseClass::Numeric(dtype)) => {
            t.numeric_dtype() == dtype
        }
        (Value::ComplexTensor(t), runmat_value::ExactHostSparseClass::Complex(dtype)) => {
            t.numeric_dtype() == dtype
        }
        (Value::SparseTensor(t), _) => {
            t.as_f32_slice().is_some()
                || t.as_f64_slice().is_some()
                || t.as_complex_f32_slice().is_some()
                || t.as_complex_f64_slice().is_some()
        }
        _ => false,
    };
    if already_native {
        return Ok(matrix.clone());
    }
    let (m, n) = matrix_dims(matrix);
    if let Value::SparseTensor(t) = matrix {
        inspect_original(matrix, false, runtime).await?;
        let mut output = sparse_factor::SparsePublication::new(
            runtime,
            runmat_value::ExactHostSparseClass::Numeric(NumericDType::F64),
            m,
            n,
            t.nnz(),
        )
        .map_err(|e| publication_error(e, runtime))?;
        let mut work = runtime.cpu_cooperation();
        for col in 0..n {
            output
                .pointer(col, t.col_ptrs[col])
                .map_err(|e| publication_error(e, runtime))?;
            for index in t.col_ptrs[col]..t.col_ptrs[col + 1] {
                let row = t.row_indices[index];
                let (real, imag) = if selected(triangle, row, col) {
                    let (real, imag) = coefficient(matrix, index)?;
                    (
                        checked_double_for(real, Some(kind))?,
                        checked_double_for(imag, Some(kind))?,
                    )
                } else {
                    (0.0, 0.0)
                };
                output
                    .entry(index, row, (real, imag))
                    .map_err(|e| publication_error(e, runtime))?;
                output
                    .checkpoint(runtime)
                    .map_err(|e| publication_error(e, runtime))?;
                work.checkpoint(1)
                    .await
                    .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
            }
            output
                .checkpoint(runtime)
                .map_err(|e| publication_error(e, runtime))?;
            work.checkpoint(1)
                .await
                .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
        }
        output
            .pointer(n, t.nnz())
            .map_err(|e| publication_error(e, runtime))?;
        return output
            .finish(runtime)
            .await
            .map_err(|e| publication_error(e, runtime));
    }
    use crate::builtins::math::linalg::ops::dense_factor::DenseScalar;
    let complex = matches!(class, runmat_value::ExactHostSparseClass::Complex(_));
    let bytes = if complex {
        HostAllocationPlan::retained_complex_tensor_bytes(NumericDType::F64, &[m, n])
    } else {
        HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[m, n])
    }
    .map_err(|e| allocation(e, runtime))?;
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, bytes).map_err(|e| allocation(e, runtime))?,
    )
    .map_err(|e| allocation(e, runtime))?;
    let mut work = runtime.cpu_cooperation();
    let mut output = if complex {
        <ComplexElement<f64> as DenseScalar>::allocate([m, n], receipt.as_ref(), runtime).await
    } else {
        <f64 as DenseScalar>::allocate([m, n], receipt.as_ref(), runtime).await
    }
    .map_err(|e| dense_error(e, runtime))?;
    let count = m
        .checked_mul(n)
        .ok_or_else(|| allocation(HostAllocationError::Overflow, runtime))?;
    for index in 0..count {
        if let Some(r) = receipt.as_ref() {
            r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
        }
        let (real, imag) = if selected(triangle, index % m, index / m) {
            let (real, imag) = coefficient(matrix, index)?;
            (
                checked_double_for(real, Some(kind))?,
                checked_double_for(imag, Some(kind))?,
            )
        } else {
            (0.0, 0.0)
        };
        if complex {
            <ComplexElement<f64> as DenseScalar>::elements_mut(&mut output)
                .expect("fresh complex owner")[index] = ComplexElement(real, imag);
        } else {
            <f64 as DenseScalar>::elements_mut(&mut output).expect("fresh real owner")[index] =
                real;
        }
        work.checkpoint(1)
            .await
            .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
    }
    if let Some(r) = receipt.as_ref() {
        r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
    }
    Ok(match output {
        runmat_value::NativeDenseFactorMatrix::Real(t) => Value::Tensor(t),
        runmat_value::NativeDenseFactorMatrix::Complex(t) => Value::ComplexTensor(t),
    })
}
fn publication_error(
    error: sparse_factor::SparsePublicationError,
    runtime: &RuntimeContext,
) -> RuntimeError {
    terminal(match error {
        sparse_factor::SparsePublicationError::Admission(e) => allocation(e, runtime),
        sparse_factor::SparsePublicationError::Execution(e) => {
            e.into_runtime_error_in_context(Some(runtime))
        }
        other => build_runtime_error(format!("decomposition sparse conversion: {other}"))
            .with_builtin(NAME)
            .with_source(other)
            .build(),
    })
}
