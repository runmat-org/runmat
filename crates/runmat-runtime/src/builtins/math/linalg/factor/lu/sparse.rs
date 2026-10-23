//! Selected public sparse LU. Native sparse factor fill is never replaced by
//! a dense coefficient snapshot; only requested output forms are published.
use crate::builtins::math::linalg::ops::sparse_factor::{
    ManagedSparseLuError, ManagedSparseScalar, PublicSparseLu, SparseLuOptions,
};
use crate::context::{
    host_allocation::{HostAllocationError, HostAllocationPlan, HostAllocationReceipt},
    RuntimeContext,
};
use crate::{BuiltinResult, RuntimeError};
use runmat_value::{NumericDType, NumericScalar, SparseTensor, Value};
mod output;
#[cfg(test)]
mod tests;
use output::{Dense, Sparse};

fn invalid(message: impl Into<String>) -> RuntimeError {
    super::lu_invalid_input(message)
}
fn allocation(runtime: &RuntimeContext, error: HostAllocationError) -> RuntimeError {
    crate::context::host_allocation::host_allocation_runtime_error(
        Some(runtime),
        "RunMat:lu:Allocation",
        "lu: sparse output admission",
        error,
    )
}
fn factor_error(runtime: &RuntimeContext, error: ManagedSparseLuError) -> RuntimeError {
    match error {
        ManagedSparseLuError::Admission(error) => allocation(runtime, error),
        ManagedSparseLuError::Execution(error) => {
            error.into_runtime_error_in_context(Some(runtime))
        }
        other => crate::build_runtime_error(format!("lu: {other}"))
            .with_identifier("RunMat:lu:InvalidInput")
            .with_builtin("lu")
            .with_source(other)
            .build(),
    }
}
fn check<T: ManagedSparseScalar>(
    factor: &PublicSparseLu<T>,
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
async fn turn<T: ManagedSparseScalar>(
    work: &mut crate::execution::CpuCooperationBudget,
    factor: &PublicSparseLu<T>,
    receipt: Option<&HostAllocationReceipt>,
    runtime: &RuntimeContext,
) -> BuiltinResult<()> {
    check(factor, receipt, runtime)?;
    work.checkpoint(1)
        .await
        .map_err(|e| e.into_runtime_error_in_context(Some(runtime)))?;
    check(factor, receipt, runtime)
}

fn options(
    args: crate::call::FunctionArgumentView<'_>,
    count: Option<usize>,
) -> BuiltinResult<(SparseLuOptions, bool)> {
    let many = count.is_some_and(|n| n >= 4);
    let mut result = SparseLuOptions {
        column_ordering: many,
        row_scaling: count == Some(5),
        unpermute_lower: count == Some(2),
        threshold: if many { [0.1, 0.001] } else { [1.0, 1.0] },
    };
    let mut vector = false;
    let mut seen_format = false;
    if args.len() > 2 {
        return Err(super::lu_invalid_argument(
            "lu: at most a sparse threshold and permutation format are accepted",
        ));
    }
    for (index, value) in args.iter().enumerate() {
        if let Some(text) = crate::builtins::common::tensor::value_to_string(value) {
            if seen_format {
                return Err(super::lu_invalid_argument(
                    "lu: at most one permutation format is accepted",
                ));
            }
            seen_format = true;
            vector = match runmat_builtins::FactorOption::parse(&text) {
                Some(runmat_builtins::FactorOption::Matrix) => false,
                Some(runmat_builtins::FactorOption::Vector) => true,
                _ => {
                    return Err(super::lu_invalid_argument(
                        "lu: permutation format must be matrix or vector",
                    ))
                }
            };
            continue;
        }
        if index != 0 {
            return Err(super::lu_invalid_argument(
                "lu: the pivot threshold precedes the output format",
            ));
        }
        let values=match value {
            Value::Num(value)=>[Some(*value),None],
            Value::Tensor(t) if matches!(t.numeric_dtype(),NumericDType::F32|NumericDType::F64) && crate::builtins::common::shape::effective_rank(t.shape())<=2 && (t.len()==1 || (many && t.len()==2))=>{
                let real=|i|match t.numeric_value_at(i) {Some(NumericScalar::F32(x))=>Some(f64::from(x)),Some(NumericScalar::F64(x))=>Some(x),_=>None};
                [real(0),if t.len()==2 {real(1)} else {None}]
            }
            _=>return Err(super::lu_invalid_argument("lu: sparse pivot thresholds require a floating scalar, or two-element vector with four or five outputs")),
        };
        for (i, value) in values.into_iter().enumerate() {
            if let Some(value) = value {
                if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                    return Err(super::lu_invalid_argument(
                        "lu: pivot thresholds must lie in [0,1]",
                    ));
                }
                result.threshold[i] = value;
            }
        }
        if !many {
            result.threshold[1] = result.threshold[0];
        }
    }
    Ok((result, vector))
}

pub(super) async fn evaluate(
    runtime: Option<&RuntimeContext>,
    source: &SparseTensor,
    args: crate::call::FunctionArgumentView<'_>,
    count: Option<usize>,
) -> BuiltinResult<Value> {
    inner(runtime, source, args, count).await.map_err(|mut e| {
        e.gpu_gather_retry = crate::GpuGatherRetry::Never;
        e
    })
}
async fn inner(
    runtime: Option<&RuntimeContext>,
    source: &SparseTensor,
    args: crate::call::FunctionArgumentView<'_>,
    count: Option<usize>,
) -> BuiltinResult<Value> {
    let owned;
    let runtime = match runtime {
        Some(runtime) => runtime,
        None => {
            if HostAllocationReceipt::policy(None).map_err(|e| {
                crate::context::host_allocation::host_allocation_runtime_error(
                    None,
                    "RunMat:lu:Allocation",
                    "lu: missing execution owner",
                    e,
                )
            })? != crate::context::host_allocation::HostAllocationPolicy::Unbounded
            {
                return Err(invalid(
                    "lu: managed sparse LU requires its execution context",
                ));
            }
            owned = RuntimeContext::new(std::rc::Rc::new(
                crate::execution::RuntimeExecutionService::new(),
            ));
            &owned
        }
    };
    let (options, vector) = options(args, count)?;
    if source.as_f32_slice().is_some() {
        run::<f32>(
            runtime,
            source,
            options,
            vector,
            count,
            NumericDType::F32,
            false,
        )
        .await
    } else if source.as_f64_slice().is_some() {
        run::<f64>(
            runtime,
            source,
            options,
            vector,
            count,
            NumericDType::F64,
            false,
        )
        .await
    } else if source.as_complex_f32_slice().is_some() {
        run::<faer::c32>(
            runtime,
            source,
            options,
            vector,
            count,
            NumericDType::F32,
            true,
        )
        .await
    } else if source.as_complex_f64_slice().is_some() {
        run::<faer::c64>(
            runtime,
            source,
            options,
            vector,
            count,
            NumericDType::F64,
            true,
        )
        .await
    } else {
        Err(invalid(
            "lu: sparse coefficients require native single or double storage",
        ))
    }
}

async fn run<T: ManagedSparseScalar>(
    runtime: &RuntimeContext,
    source: &SparseTensor,
    options: SparseLuOptions,
    vector: bool,
    count: Option<usize>,
    dtype: NumericDType,
    complex: bool,
) -> BuiltinResult<Value> {
    let factor = PublicSparseLu::<T>::factor(source, options, runtime)
        .await
        .map_err(|e| factor_error(runtime, e))?;
    let (m, n) = factor.dimensions();
    let k = m.min(n);
    let requested = count.unwrap_or(1);
    let packed = requested == 1;
    if requested == 0 {
        factor
            .check_owner(runtime)
            .map_err(|e| allocation(runtime, e))?;
        return Ok(Value::output_list(Vec::new()));
    }
    let mut work = runtime.cpu_cooperation();
    let mut lnnz = 0usize;
    let mut unnz = 0usize;
    for col in 0..n {
        let (rows, _) = factor
            .upper_column(col, runtime)
            .map_err(|e| factor_error(runtime, e))?;
        unnz = unnz
            .checked_add(rows.len())
            .ok_or_else(|| invalid("lu: upper nnz overflow"))?;
        if col < k {
            for &row in factor
                .lower_column(col, runtime)
                .map_err(|e| factor_error(runtime, e))?
                .0
            {
                if !packed || row > col {
                    lnnz = lnnz
                        .checked_add(1)
                        .ok_or_else(|| invalid("lu: lower nnz overflow"))?;
                }
                turn(&mut work, &factor, None, runtime).await?;
            }
        }
        turn(&mut work, &factor, None, runtime).await?;
    }
    let packed_nnz = lnnz
        .checked_add(unnz)
        .ok_or_else(|| invalid("lu: packed nnz overflow"))?;
    // SparsePublication independently admits each exact sparse owner. This
    // receipt covers only the result-list and dense index vectors, never a
    // second reservation for those same sparse payloads.
    let mut retained = 0u64;
    for size in [
        if requested >= 3 { Some(m) } else { None },
        if requested >= 4 { Some(n) } else { None },
    ]
    .into_iter()
    .flatten()
    {
        let bytes = if vector {
            HostAllocationPlan::retained_tensor_bytes(NumericDType::F64, &[size, 1])
                .map_err(|e| allocation(runtime, e))?
        } else {
            0
        };
        retained = retained
            .checked_add(bytes)
            .ok_or_else(|| invalid("lu: permutation bytes overflow"))?;
    }
    if count.is_some() {
        retained = retained
            .checked_add(
                super::super::iterative::output::retained_list_bytes(requested)
                    .map_err(|e| allocation(runtime, e))?,
            )
            .ok_or_else(|| invalid("lu: list bytes overflow"))?;
    }
    let receipt = HostAllocationReceipt::admit(
        Some(runtime),
        HostAllocationPlan::new(0, retained).map_err(|e| allocation(runtime, e))?,
    )
    .map_err(|e| allocation(runtime, e))?;
    check(&factor, receipt.as_ref(), runtime)?;
    let mut outputs: [Option<Value>; 5] = [None, None, None, None, None];
    let mut l = Sparse::new(
        runtime,
        receipt.as_ref(),
        dtype,
        complex,
        m,
        if packed { n } else { k },
        if packed { packed_nnz } else { lnnz },
    )?;
    l.ptr(0, 0)?;
    let mut written = 0;
    for col in 0..if packed { n } else { k } {
        if packed {
            let (rows, values) = factor
                .upper_column(col, runtime)
                .map_err(|e| factor_error(runtime, e))?;
            for (&row, &value) in rows.iter().zip(values) {
                l.entry(written, row, value.parts(), dtype, complex)?;
                written += 1;
                turn(&mut work, &factor, receipt.as_ref(), runtime).await?;
            }
        }
        if col < k {
            let (rows, values) = factor
                .lower_column(col, runtime)
                .map_err(|e| factor_error(runtime, e))?;
            for (&row, &value) in rows.iter().zip(values) {
                if !packed || row > col {
                    l.entry(written, row, value.parts(), dtype, complex)?;
                    written += 1;
                }
                turn(&mut work, &factor, receipt.as_ref(), runtime).await?;
            }
        }
        l.ptr(col + 1, written)?;
    }
    outputs[0] = Some(
        l.finish(runtime, || check(&factor, receipt.as_ref(), runtime))
            .await?,
    );
    if !packed {
        let mut u = Sparse::new(runtime, receipt.as_ref(), dtype, complex, k, n, unnz)?;
        u.ptr(0, 0)?;
        let mut written = 0;
        for col in 0..n {
            let (rows, values) = factor
                .upper_column(col, runtime)
                .map_err(|e| factor_error(runtime, e))?;
            for (&row, &value) in rows.iter().zip(values) {
                u.entry(written, row, value.parts(), dtype, complex)?;
                written += 1;
                turn(&mut work, &factor, receipt.as_ref(), runtime).await?;
            }
            u.ptr(col + 1, written)?;
        }
        outputs[1] = Some(
            u.finish(runtime, || check(&factor, receipt.as_ref(), runtime))
                .await?,
        );
    }
    for slot in 2..requested.min(4) {
        let p = if slot == 2 {
            factor.row_permutation(runtime)
        } else {
            factor.column_permutation(runtime)
        }
        .map_err(|e| allocation(runtime, e))?;
        if vector {
            let mut output = Dense::new(
                runtime,
                receipt.as_ref(),
                NumericDType::F64,
                false,
                p.len(),
                1,
            )?;
            for (index, value) in p.iter().enumerate() {
                output.set(index, (value + 1) as f64, 0.0, runtime)?;
                turn(&mut work, &factor, receipt.as_ref(), runtime).await?;
            }
            outputs[slot] = Some(output.finish(runtime)?);
        } else {
            let indices = if slot == 2 {
                factor
                    .row_inverse(runtime)
                    .map_err(|e| allocation(runtime, e))?
            } else {
                p
            };
            let mut output = Sparse::new(
                runtime,
                receipt.as_ref(),
                NumericDType::F64,
                false,
                p.len(),
                p.len(),
                p.len(),
            )?;
            for (column, row) in indices.iter().enumerate() {
                output.ptr(column, column)?;
                output.entry(column, row, (1.0, 0.0), NumericDType::F64, false)?;
                turn(&mut work, &factor, receipt.as_ref(), runtime).await?;
            }
            output.ptr(p.len(), p.len())?;
            outputs[slot] = Some(
                output
                    .finish(runtime, || check(&factor, receipt.as_ref(), runtime))
                    .await?,
            );
        }
    }
    if requested == 5 {
        let mut output = Sparse::new(runtime, receipt.as_ref(), dtype, false, m, m, m)?;
        for (row, &value) in factor
            .scaling(runtime)
            .map_err(|e| allocation(runtime, e))?
            .iter()
            .enumerate()
        {
            output.ptr(row, row)?;
            output.entry(row, row, (value.parts().0, 0.0), dtype, false)?;
            turn(&mut work, &factor, receipt.as_ref(), runtime).await?;
        }
        output.ptr(m, m)?;
        outputs[4] = Some(
            output
                .finish(runtime, || check(&factor, receipt.as_ref(), runtime))
                .await?,
        );
    }
    check(&factor, receipt.as_ref(), runtime)?;
    if count.is_none() {
        return Ok(outputs[0].take().expect("selected packed factor"));
    }
    if let Some(receipt) = receipt.as_ref() {
        let permit = receipt
            .output_permit(runtime)
            .map_err(|e| allocation(runtime, e))?;
        super::super::iterative::output::initialize_list(requested, outputs, &permit)
            .map_err(|e| allocation(runtime, e.into()))
    } else {
        Ok(Value::output_list(
            outputs
                .into_iter()
                .take(requested)
                .map(|v| v.expect("selected LU output"))
                .collect::<Vec<_>>(),
        ))
    }
}
