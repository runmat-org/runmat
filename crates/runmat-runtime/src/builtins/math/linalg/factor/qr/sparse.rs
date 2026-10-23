//! Public sparse QR transport. Coefficients remain CSC; explicit Q/C outputs
//! are independently admitted dense results, not dense coefficient snapshots.
use crate::builtins::math::linalg::ops::sparse_factor::{
    SparseCompleteQrError, SparseQrDecomposition, SparseQrScalar,
};
use crate::context::{
    host_allocation::{
        HostAllocationError, HostAllocationPlan, HostAllocationReceipt, HostTypedWorkspace,
    },
    RuntimeContext,
};
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{NumericDType, NumericScalar, SparseCscValidation, SparseTensor, Value};
#[cfg(test)]
mod tests;

fn allocation(runtime: &RuntimeContext, error: HostAllocationError) -> RuntimeError {
    let mut result = crate::context::host_allocation::host_allocation_runtime_error(
        Some(runtime),
        "RunMat:qr:Allocation",
        "qr: sparse factor output admission",
        error,
    );
    result.gpu_gather_retry = crate::GpuGatherRetry::Never;
    result
}
fn factor_error(runtime: &RuntimeContext, error: SparseCompleteQrError) -> RuntimeError {
    let mut result = match error {
        SparseCompleteQrError::Admission(error) => allocation(runtime, error),
        SparseCompleteQrError::Execution(error) => {
            error.into_runtime_error_in_context(Some(runtime))
        }
        other => crate::build_runtime_error(format!("qr: {other}"))
            .with_identifier("RunMat:qr:InvalidInput")
            .with_builtin("qr")
            .with_source(other)
            .build(),
    };
    result.gpu_gather_retry = crate::GpuGatherRetry::Never;
    result
}
fn invalid(message: impl Into<String>) -> RuntimeError {
    super::qr_invalid_input(message)
}
fn check<T: SparseQrScalar>(
    factor: &SparseQrDecomposition<T>,
    receipt: Option<&HostAllocationReceipt>,
    runtime: &RuntimeContext,
) -> BuiltinResult<()> {
    factor
        .check_owner(runtime)
        .map_err(|e| allocation(runtime, e))?;
    if let Some(receipt) = receipt {
        receipt
            .checkpoint(runtime)
            .map_err(|e| allocation(runtime, e))?;
    }
    Ok(())
}
async fn turn<T: SparseQrScalar>(
    work: &mut crate::execution::CpuCooperationBudget,
    factor: &SparseQrDecomposition<T>,
    receipt: Option<&HostAllocationReceipt>,
    runtime: &RuntimeContext,
    units: u64,
) -> BuiltinResult<()> {
    check(factor, receipt, runtime)?;
    work.checkpoint(units)
        .await
        .map_err(|e| e.into_runtime_error_in_context(Some(runtime)))?;
    check(factor, receipt, runtime)
}

pub(super) async fn evaluate(
    runtime: Option<&RuntimeContext>,
    a: &SparseTensor,
    args: crate::call::FunctionArgumentView<'_>,
    count: Option<usize>,
) -> BuiltinResult<Value> {
    evaluate_inner(runtime, a, args, count)
        .await
        .map_err(|mut error| {
            // This selected sparse route owns every argument read, factorization
            // and output attempt. In particular, an option owner read must not be
            // replayed when a later option is invalid.
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        })
}

async fn evaluate_inner(
    runtime: Option<&RuntimeContext>,
    a: &SparseTensor,
    args: crate::call::FunctionArgumentView<'_>,
    count: Option<usize>,
) -> BuiltinResult<Value> {
    let owned;
    let runtime = match runtime {
        Some(runtime) => runtime,
        None => {
            // Ordinary context-free embedding is supported, but never borrow
            // another session's finite allocation or execution services.
            if HostAllocationReceipt::policy(None).map_err(|e| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    None,
                    "RunMat:qr:Allocation",
                    "qr: missing execution owner",
                    e,
                )
            })? != crate::context::host_allocation::HostAllocationPolicy::Unbounded
            {
                return Err(invalid(
                    "qr: managed sparse QR requires its execution context",
                ));
            }
            owned = RuntimeContext::new(std::rc::Rc::new(
                crate::execution::RuntimeExecutionService::new(),
            ));
            &owned
        }
    };
    let rhs = match args.first() {
        Some(value)
            if matches!(
                value,
                Value::Num(_)
                    | Value::Complex(_, _)
                    | Value::Tensor(_)
                    | Value::ComplexTensor(_)
                    | Value::SparseTensor(_)
            ) && !super::is_zero_scalar(value).await? =>
        {
            Some(value)
        }
        _ => None,
    };
    if rhs.is_some() && count.is_none_or(|n| n < 2) {
        return Err(invalid(
            "qr: sparse two-input QR requires two or three outputs",
        ));
    }
    let options = super::parse_options(
        args.range(usize::from(rhs.is_some())..)
            .expect("bounded QR options"),
    )
    .await?;
    if count != Some(3) && options.format.requires_permutation_output() {
        return Err(super::qr_invalid_argument(
            "qr: matrix and vector permutation options require three outputs",
        ));
    }
    if a.as_f32_slice().is_some() {
        run::<f32>(runtime, a, rhs, options, count, NumericDType::F32, false).await
    } else if a.as_f64_slice().is_some() {
        run::<f64>(runtime, a, rhs, options, count, NumericDType::F64, false).await
    } else if a.as_complex_f32_slice().is_some() {
        run::<faer::c32>(runtime, a, rhs, options, count, NumericDType::F32, true).await
    } else if a.as_complex_f64_slice().is_some() {
        run::<faer::c64>(runtime, a, rhs, options, count, NumericDType::F64, true).await
    } else {
        Err(invalid(
            "qr: sparse coefficients require single or double native storage",
        ))
    }
}

fn shape(value: &Value) -> Option<(usize, usize, bool)> {
    match value {
        Value::Num(_) => Some((1, 1, false)),
        Value::Complex(_, _) => Some((1, 1, true)),
        Value::Tensor(t)
            if matches!(t.numeric_dtype(), NumericDType::F32 | NumericDType::F64)
                && t.shape().len() == 2 =>
        {
            Some((t.shape()[0], t.shape()[1], false))
        }
        Value::ComplexTensor(t)
            if matches!(t.numeric_dtype(), NumericDType::F32 | NumericDType::F64)
                && t.shape().len() == 2 =>
        {
            Some((t.shape()[0], t.shape()[1], true))
        }
        Value::SparseTensor(t) if t.as_f32_slice().is_some() || t.as_f64_slice().is_some() => {
            Some((t.rows, t.cols, false))
        }
        Value::SparseTensor(t)
            if t.as_complex_f32_slice().is_some() || t.as_complex_f64_slice().is_some() =>
        {
            Some((t.rows, t.cols, true))
        }
        _ => None,
    }
}
fn parts(value: &Value, row: usize, col: usize) -> BuiltinResult<(f64, f64)> {
    let (m, _, _) =
        shape(value).ok_or_else(|| invalid("qr: B must be a native floating matrix"))?;
    let result = match value {
        Value::Num(value) => (*value, 0.0),
        Value::Complex(re, im) => (*re, *im),
        Value::Tensor(t) => match t.numeric_value_at(col * m + row) {
            Some(NumericScalar::F32(v)) => (f64::from(v), 0.0),
            Some(NumericScalar::F64(v)) => (v, 0.0),
            _ => return Err(invalid("qr: invalid B storage")),
        },
        Value::ComplexTensor(t) => {
            if let Some(v) = t.as_f32_slice() {
                let x = v[col * m + row];
                (f64::from(x.0), f64::from(x.1))
            } else {
                let x = t
                    .as_f64_slice()
                    .ok_or_else(|| invalid("qr: invalid complex B"))?[col * m + row];
                (x.0, x.1)
            }
        }
        Value::SparseTensor(t) => {
            let start = t.col_ptrs[col];
            let rows = &t.row_indices[start..t.col_ptrs[col + 1]];
            match rows.binary_search(&row) {
                Err(_) => (0.0, 0.0),
                Ok(index) => {
                    let i = start + index;
                    if let Some(v) = t.as_complex_f32_slice() {
                        (f64::from(v[i].0), f64::from(v[i].1))
                    } else if let Some(v) = t.as_complex_f64_slice() {
                        (v[i].0, v[i].1)
                    } else {
                        match t.numeric_value_at(i) {
                            Some(NumericScalar::F32(v)) => (f64::from(v), 0.0),
                            Some(NumericScalar::F64(v)) => (v, 0.0),
                            _ => return Err(invalid("qr: invalid sparse B")),
                        }
                    }
                }
            }
        }
        _ => return Err(invalid("qr: invalid B storage")),
    };
    if !result.0.is_finite() || !result.1.is_finite() {
        return Err(invalid("qr: B must contain finite values"));
    }
    Ok(result)
}

mod output;
use output::{sparse_bytes, Column, Dense, Sparse};

async fn run<T: SparseQrScalar>(
    runtime: &RuntimeContext,
    a: &SparseTensor,
    b: Option<&Value>,
    options: super::QrOptions,
    count: Option<usize>,
    dtype: NumericDType,
    complex: bool,
) -> BuiltinResult<Value> {
    let (m, n) = (a.rows, a.cols);
    let pivot = count == Some(3);
    let economy = matches!(options.mode, super::QrMode::Economy);
    let height = if economy { m.min(n) } else { m };
    let want_q = count.is_some_and(|c| c >= 2);
    let (qrows, qcols, qcomplex) = if let Some(b) = b {
        let (rows, cols, bc) =
            shape(b).ok_or_else(|| invalid("qr: B must be a native floating matrix"))?;
        if rows != m {
            return Err(invalid("qr: B row count must equal S row count"));
        }
        (height, cols, complex || bc)
    } else {
        (m, if economy { m.min(n) } else { m }, complex)
    };
    if let Some(Value::SparseTensor(b)) = b {
        let mut validator = SparseCscValidation::new(
            b.rows,
            b.cols,
            b.col_ptrs.clone(),
            b.row_indices.clone(),
            b.nnz(),
        )
        .map_err(invalid)?;
        let mut work = runtime.cpu_cooperation();
        loop {
            let done = validator
                .advance(std::num::NonZeroUsize::new(256).expect("nonzero validation work"))
                .map_err(invalid)?;
            work.checkpoint(256)
                .await
                .map_err(|e| e.into_runtime_error_in_context(Some(runtime)))?;
            if done {
                break;
            }
        }
    }
    let factor = SparseQrDecomposition::<T>::factor(a, pivot, runtime)
        .await
        .map_err(|e| factor_error(runtime, e))?;
    if count == Some(0) {
        factor
            .check_owner(runtime)
            .map_err(|e| allocation(runtime, e))?;
        return Ok(Value::output_list(Vec::new()));
    }
    let mut nnz = 0usize;
    let mut work = runtime.cpu_cooperation();
    for col in 0..n {
        let (rows, _) = factor
            .r_column(col, runtime)
            .map_err(|e| factor_error(runtime, e))?;
        for &row in rows {
            if row < height {
                nnz = nnz
                    .checked_add(1)
                    .ok_or_else(|| invalid("qr: nnz overflow"))?;
            }
            turn(&mut work, &factor, None, runtime, 1).await?;
        }
    }
    let mut retained =
        sparse_bytes(dtype, complex, height, n, nnz).map_err(|e| allocation(runtime, e))?;
    if want_q {
        retained = retained
            .checked_add(
                if qcomplex {
                    HostAllocationPlan::retained_complex_tensor_bytes(dtype, &[qrows, qcols])
                } else {
                    HostAllocationPlan::retained_tensor_bytes(dtype, &[qrows, qcols])
                }
                .map_err(|e| allocation(runtime, e))?,
            )
            .ok_or_else(|| invalid("qr: output bytes overflow"))?;
    }
    let vector = matches!(options.format.pivot_mode(), super::PivotMode::Vector);
    if pivot {
        retained = retained
            .checked_add(
                if vector {
                    HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[n, 1])
                } else {
                    sparse_bytes(NumericDType::F64, false, n, n, n)
                }
                .map_err(|e| allocation(runtime, e))?,
            )
            .ok_or_else(|| invalid("qr: permutation bytes overflow"))?;
    }
    if let Some(c @ 1..=3) = count {
        retained = retained
            .checked_add(
                super::super::iterative::output::retained_list_bytes(c)
                    .map_err(|e| allocation(runtime, e))?,
            )
            .ok_or_else(|| invalid("qr: list bytes overflow"))?;
    }
    let lanes = if !complex && qcomplex { 2usize } else { 1 };
    let needs_column = want_q && qrows != 0 && qcols != 0;
    let column_len = if needs_column {
        m.checked_mul(lanes)
            .ok_or_else(|| invalid("qr: workspace dimensions overflow"))?
    } else {
        0
    };
    let scratch = if needs_column {
        HostTypedWorkspace::<T>::quoted_bytes(column_len).map_err(|e| allocation(runtime, e))?
    } else {
        0
    };
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(scratch, retained).map_err(|e| allocation(runtime, e))?,
    )
    .map_err(|e| allocation(runtime, e))?;
    check(&factor, receipt.as_ref(), runtime)?;
    let mut r = Sparse::new(runtime, receipt.as_ref(), dtype, complex, height, n, nnz)?;
    r.ptr(0, 0)?;
    let mut written = 0;
    for col in 0..n {
        let (rows, values) = factor
            .r_column(col, runtime)
            .map_err(|e| factor_error(runtime, e))?;
        for (&row, &value) in rows.iter().zip(values) {
            if row < height {
                r.entry(written, row, value.parts())?;
                written += 1;
            }
            turn(&mut work, &factor, receipt.as_ref(), runtime, 1).await?;
        }
        r.ptr(col + 1, written)?;
    }
    let r = r
        .finish(runtime, || check(&factor, receipt.as_ref(), runtime))
        .await?;
    let mut q = None;
    if want_q {
        let mut output = Dense::new(runtime, receipt.as_ref(), dtype, qcomplex, qrows, qcols)?;
        if needs_column {
            let mut column = if let Some(receipt) = &receipt {
                Column::Managed(
                    receipt
                        .temporary_typed(runtime, column_len)
                        .map_err(|e| allocation(runtime, e))?,
                )
            } else {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(column_len)
                    .map_err(|_| invalid("qr: workspace allocation failed"))?;
                for _ in 0..column_len {
                    values.push(T::default());
                    turn(&mut work, &factor, receipt.as_ref(), runtime, 1).await?;
                }
                Column::Ordinary(values)
            };
            for col in 0..qcols {
                for row in 0..m {
                    let (re, im) = if let Some(b) = b {
                        parts(b, row, col)?
                    } else {
                        (if row == col { 1.0 } else { 0.0 }, 0.0)
                    };
                    column.slice()[row] = T::from_parts(re, if complex { im } else { 0.0 })
                        .ok_or_else(|| invalid("qr: B exceeds native coefficient range"))?;
                    if lanes == 2 {
                        column.slice()[m + row] = T::from_parts(im, 0.0)
                            .ok_or_else(|| invalid("qr: B exceeds native coefficient range"))?;
                    }
                    turn(&mut work, &factor, receipt.as_ref(), runtime, 1).await?;
                }
                factor
                    .apply_q(&mut column.slice()[..m], 1, b.is_some(), runtime)
                    .await
                    .map_err(|e| factor_error(runtime, e))?;
                if lanes == 2 {
                    factor
                        .apply_q(&mut column.slice()[m..], 1, true, runtime)
                        .await
                        .map_err(|e| factor_error(runtime, e))?;
                }
                check(&factor, receipt.as_ref(), runtime)?;
                for row in 0..qrows {
                    let (re, mut im) = column.slice()[row].parts();
                    if lanes == 2 {
                        im = column.slice()[m + row].parts().0;
                    }
                    output.set(col * qrows + row, re, im, runtime)?;
                    turn(&mut work, &factor, receipt.as_ref(), runtime, 1).await?;
                }
            }
        }
        q = Some(output.finish(runtime)?);
    }
    let permutation = if pivot {
        let p = factor
            .permutation(runtime)
            .map_err(|e| allocation(runtime, e))?;
        if vector {
            let mut out = Dense::new(runtime, receipt.as_ref(), NumericDType::F64, false, n, 1)?;
            for (i, &row) in p.iter().enumerate() {
                out.set(i, (row + 1) as f64, 0.0, runtime)?;
                turn(&mut work, &factor, receipt.as_ref(), runtime, 1).await?;
            }
            Some(out.finish(runtime)?)
        } else {
            let mut out =
                Sparse::new(runtime, receipt.as_ref(), NumericDType::F64, false, n, n, n)?;
            for (i, &row) in p.iter().enumerate() {
                out.ptr(i, i)?;
                out.entry(i, row, (1.0, 0.0))?;
                turn(&mut work, &factor, receipt.as_ref(), runtime, 1).await?;
            }
            out.ptr(n, n)?;
            Some(
                out.finish(runtime, || check(&factor, receipt.as_ref(), runtime))
                    .await?,
            )
        }
    } else {
        None
    };
    factor
        .check_owner(runtime)
        .map_err(|e| allocation(runtime, e))?;
    if let Some(receipt) = &receipt {
        receipt
            .checkpoint(runtime)
            .map_err(|e| allocation(runtime, e))?;
    }
    match count {
        None => Ok(r),
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(c @ 1..=3) => {
            let outputs = if c == 1 {
                [Some(r), None, None, None, None]
            } else {
                [q, Some(r), permutation, None, None]
            };
            if let Some(receipt) = &receipt {
                let permit = receipt
                    .output_permit(runtime)
                    .map_err(|e| allocation(runtime, e))?;
                super::super::iterative::output::initialize_list(c, outputs, &permit)
                    .map_err(|e| allocation(runtime, e.into()))
            } else {
                Ok(Value::output_list(
                    outputs
                        .into_iter()
                        .take(c)
                        .map(|v| v.expect("selected output"))
                        .collect::<Vec<_>>(),
                ))
            }
        }
        _ => Err(invalid("qr: at most three outputs")),
    }
}
