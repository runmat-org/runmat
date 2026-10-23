//! Selected decomposition inspection owner.
use super::*;
fn same(a: NumericScalar, b: NumericScalar) -> bool {
    if a.is_zero() && b.is_zero() {
        return true;
    }
    a == b
}
fn conjugate_same(a: (NumericScalar, NumericScalar), b: (NumericScalar, NumericScalar)) -> bool {
    if !same(a.0, b.0) {
        return false;
    }
    match (a.1.into_int_value(), b.1.into_int_value()) {
        (Some(a), Some(b)) => exact_integer(a) == -exact_integer(b),
        _ => match (a.1, b.1) {
            (NumericScalar::F32(a), NumericScalar::F32(b)) => a == -b,
            (NumericScalar::F64(a), NumericScalar::F64(b)) => a == -b,
            (a, b) => a.is_zero() && b.is_zero(),
        },
    }
}

pub(in crate::builtins::math::linalg::factor::decomposition) fn selected(
    triangle: Option<TriangularFlag>,
    row: usize,
    column: usize,
) -> bool {
    match triangle {
        None => true,
        Some(TriangularFlag::Lower) => row >= column,
        Some(TriangularFlag::Upper) => row <= column,
    }
}
pub(in crate::builtins::math::linalg::factor::decomposition) async fn selected_real(
    matrix: &Value,
    triangle: Option<TriangularFlag>,
    runtime: &RuntimeContext,
) -> BuiltinResult<bool> {
    let (m, n) = matrix_dims(matrix);
    let mut real = true;
    let mut work = runtime.cpu_cooperation();
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, 0).map_err(|e| allocation(e, runtime))?,
    )
    .map_err(|e| allocation(e, runtime))?;
    if let Value::SparseTensor(t) = matrix {
        for col in 0..n {
            for index in t.col_ptrs[col]..t.col_ptrs[col + 1] {
                if let Some(r) = receipt.as_ref() {
                    r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
                }
                if selected(triangle, t.row_indices[index], col) {
                    real &= coefficient(matrix, index)?.1.is_zero();
                }
                work.checkpoint(1)
                    .await
                    .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
            }
            work.checkpoint(1)
                .await
                .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
        }
    } else {
        let count = m
            .checked_mul(n)
            .ok_or_else(|| allocation(HostAllocationError::Overflow, runtime))?;
        for index in 0..count {
            if let Some(r) = receipt.as_ref() {
                r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
            }
            if selected(triangle, index % m, index / m) {
                real &= coefficient(matrix, index)?.1.is_zero();
            }
            work.checkpoint(1)
                .await
                .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
        }
    }
    if let Some(r) = receipt.as_ref() {
        r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
    }
    work.checkpoint(0)
        .await
        .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
    Ok(real)
}
pub(in crate::builtins::math::linalg::factor::decomposition) async fn check_factor_recipe(
    matrix: &Value,
    triangle: Option<TriangularFlag>,
    runtime: &RuntimeContext,
) -> BuiltinResult<()> {
    let (m, n) = matrix_dims(matrix);
    let mut work = runtime.cpu_cooperation();
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, 0).map_err(|e| allocation(e, runtime))?,
    )
    .map_err(|e| allocation(e, runtime))?;
    let check = |index| -> BuiltinResult<()> {
        let (real, imag) = coefficient(matrix, index)?;
        if !checked_double(real)?.is_finite() || !checked_double(imag)?.is_finite() {
            return Err(invalid(
                "decomposition factor recipe requires finite selected coefficients",
            ));
        }
        Ok(())
    };
    if let Value::SparseTensor(t) = matrix {
        for col in 0..n {
            for index in t.col_ptrs[col]..t.col_ptrs[col + 1] {
                if let Some(r) = receipt.as_ref() {
                    r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
                }
                if selected(triangle, t.row_indices[index], col) {
                    check(index)?;
                }
                work.checkpoint(1)
                    .await
                    .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
            }
            work.checkpoint(1)
                .await
                .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
        }
    } else {
        let count = m
            .checked_mul(n)
            .ok_or_else(|| allocation(HostAllocationError::Overflow, runtime))?;
        for index in 0..count {
            if let Some(r) = receipt.as_ref() {
                r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
            }
            if selected(triangle, index % m, index / m) {
                check(index)?;
            }
            work.checkpoint(1)
                .await
                .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
        }
    }
    if let Some(r) = receipt.as_ref() {
        r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
    }
    work.checkpoint(0)
        .await
        .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
    Ok(())
}
pub(in crate::builtins::math::linalg::factor::decomposition) async fn inspect_original(
    value: &Value,
    hermitian: bool,
    runtime: &RuntimeContext,
) -> BuiltinResult<NativeMatrixFacts> {
    let native = matches!(value,Value::Tensor(t) if t.as_f32_slice().is_some()||t.as_f64_slice().is_some())
        || matches!(value,Value::ComplexTensor(t) if t.as_f32_slice().is_some()||t.as_f64_slice().is_some())
        || matches!(value,Value::SparseTensor(t) if t.as_f32_slice().is_some()||t.as_f64_slice().is_some()||t.as_complex_f32_slice().is_some()||t.as_complex_f64_slice().is_some());
    if native {
        return sparse_factor::inspect_native_matrix(value, hermitian, runtime)
            .await
            .map_err(|e| retained_error(e, runtime));
    }
    let (m, n) = matrix_dims(value);
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, 0).map_err(|e| allocation(e, runtime))?,
    )
    .map_err(|e| allocation(e, runtime))?;
    let mut work = runtime.cpu_cooperation();
    work.checkpoint(0)
        .await
        .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
    let mut facts = NativeMatrixFacts {
        diagonal: true,
        upper: true,
        lower: true,
        real: true,
        hermitian: m == n,
    };
    if let Value::SparseTensor(t) = value {
        let mut proof = runmat_value::SparseCscValidation::new(
            m,
            n,
            t.col_ptrs.clone(),
            t.row_indices.clone(),
            t.nnz(),
        )
        .map_err(invalid)?;
        loop {
            if let Some(r) = receipt.as_ref() {
                r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
            }
            let done = proof
                .advance(std::num::NonZeroUsize::new(1).expect("one"))
                .map_err(invalid)?;
            work.checkpoint(1)
                .await
                .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
            if done {
                break;
            }
        }
        for col in 0..n {
            for index in t.col_ptrs[col]..t.col_ptrs[col + 1] {
                if let Some(r) = receipt.as_ref() {
                    r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
                }
                let row = t.row_indices[index];
                let v = coefficient(value, index)?;
                facts.real &= v.1.is_zero();
                if !v.0.is_zero() || !v.1.is_zero() {
                    facts.diagonal &= row == col;
                    facts.upper &= row <= col;
                    facts.lower &= row >= col;
                }
                if hermitian && m == n {
                    let mut low = t.col_ptrs[row];
                    let mut high = t.col_ptrs[row + 1];
                    let mut opposite = None;
                    while low < high {
                        let mid = low + (high - low) / 2;
                        match t.row_indices[mid].cmp(&col) {
                            std::cmp::Ordering::Less => low = mid + 1,
                            std::cmp::Ordering::Greater => high = mid,
                            std::cmp::Ordering::Equal => {
                                opposite = Some(coefficient(value, mid)?);
                                break;
                            }
                        }
                        work.checkpoint(1).await.map_err(|e| {
                            terminal(e.into_runtime_error_in_context(Some(runtime)))
                        })?;
                    }
                    facts.hermitian &= conjugate_same(
                        v,
                        opposite.unwrap_or((NumericScalar::F64(0.0), NumericScalar::F64(0.0))),
                    );
                }
                work.checkpoint(1)
                    .await
                    .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
            }
            work.checkpoint(1)
                .await
                .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
        }
        if let Some(r) = receipt.as_ref() {
            r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
        }
        return Ok(facts);
    }
    let len = match value {
        Value::Tensor(t) => t.len(),
        Value::ComplexTensor(t) => t.len(),
        Value::LogicalArray(t) => t.data.len(),
        _ => return Err(invalid("expected normalized numeric matrix")),
    };
    for index in 0..len {
        if let Some(r) = receipt.as_ref() {
            r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
        }
        let row = index % m;
        let col = index / m;
        let v = coefficient(value, index)?;
        facts.real &= v.1.is_zero();
        if !v.0.is_zero() || !v.1.is_zero() {
            facts.diagonal &= row == col;
            facts.upper &= row <= col;
            facts.lower &= row >= col;
        }
        if hermitian && m == n {
            facts.hermitian &= conjugate_same(v, coefficient(value, col + row * m)?);
        }
        work.checkpoint(1)
            .await
            .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
        if let Some(r) = receipt.as_ref() {
            r.checkpoint(runtime).map_err(|e| allocation(e, runtime))?;
        }
    }
    work.checkpoint(0)
        .await
        .map_err(|e| terminal(e.into_runtime_error_in_context(Some(runtime))))?;
    Ok(facts)
}
