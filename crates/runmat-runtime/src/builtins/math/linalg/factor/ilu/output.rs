use super::*;

pub(super) async fn evaluate(
    sparse: SparseTensor,
    options: Options,
    outputs: usize,
    runtime: &RuntimeContext,
) -> BuiltinResult<PublishedValues> {
    match sparse.numeric_dtype() {
        Some(NumericDType::F32) if sparse.as_complex_f32_slice().is_some() => {
            evaluate_slots::<faer::c32>(sparse, options, outputs, runtime).await
        }
        Some(NumericDType::F64) if sparse.as_complex_f64_slice().is_some() => {
            evaluate_slots::<faer::c64>(sparse, options, outputs, runtime).await
        }
        Some(NumericDType::F32) => evaluate_slots::<f32>(sparse, options, outputs, runtime).await,
        Some(NumericDType::F64) => evaluate_slots::<f64>(sparse, options, outputs, runtime).await,
        _ => Err(error(&ILU_ERROR_INVALID_INPUT, None)),
    }
}

pub(super) struct PublishedValues {
    pub(super) values: [Option<Value>; 3],
}

async fn evaluate_slots<T: ManagedSparseScalar>(
    sparse: SparseTensor,
    options: Options,
    outputs: usize,
    runtime: &RuntimeContext,
) -> BuiltinResult<PublishedValues> {
    let (drop_tolerance, replace_zero_diagonal, pivot_threshold) = match options.kind {
        IncompleteLuKind::NoFill => (0.0, false, 0.0),
        IncompleteLuKind::Crout => (options.droptol, false, 0.0),
        IncompleteLuKind::ThresholdPivoting => (options.droptol, options.udiag, options.thresh),
    };
    let transpose_source = matches!(options.kind, IncompleteLuKind::ThresholdPivoting)
        && matches!(options.modified, ModifiedIncompleteLu::Column);
    let core_modified = if transpose_source {
        ModifiedIncompleteLu::Row
    } else {
        options.modified
    };
    let factor = ManagedIncompleteLu::<T>::factor(
        &sparse,
        IncompleteLuOptions {
            kind: options.kind,
            drop_tolerance,
            modified: core_modified,
            replace_zero_diagonal,
            pivot_threshold,
            transpose_source,
        },
        runtime,
    )
    .await
    .map_err(|e| factor_error(e, runtime))?;
    let (mut lower, mut upper) = if transpose_source {
        factor
            .project_transposed(runtime)
            .await
            .map_err(|e| factor_error(e, runtime))?
    } else {
        (
            factor
                .project_csc(IncompleteLuPart::Lower, runtime)
                .await
                .map_err(|e| factor_error(e, runtime))?,
            factor
                .project_csc(IncompleteLuPart::Upper, runtime)
                .await
                .map_err(|e| factor_error(e, runtime))?,
        )
    };
    if outputs == 2 && matches!(options.kind, IncompleteLuKind::ThresholdPivoting) {
        if transpose_source {
            lower = factor
                .fold_row_permutation(lower, runtime)
                .await
                .map_err(|e| factor_error(e, runtime))?;
        } else {
            upper = factor
                .fold_column_permutation(&upper, runtime)
                .await
                .map_err(|e| factor_error(e, runtime))?;
        }
    }
    if outputs == 1 {
        return Ok(PublishedValues {
            values: [
                Some(publish_combined(&factor, &lower, &upper, &sparse, runtime).await?),
                None,
                None,
            ],
        });
    }
    Ok(PublishedValues {
        values: [
            Some(publish(&factor, &lower, &sparse, runtime).await?),
            Some(publish(&factor, &upper, &sparse, runtime).await?),
            if outputs == 3 {
                Some(publish_permutation(&factor, transpose_source, runtime).await?)
            } else {
                None
            },
        ],
    })
}

#[cfg(test)]
pub(super) async fn evaluate_typed<T: ManagedSparseScalar>(
    sparse: SparseTensor,
    options: Options,
    outputs: usize,
    runtime: &RuntimeContext,
) -> BuiltinResult<Vec<Value>> {
    let mut published = evaluate_slots::<T>(sparse, options, outputs, runtime).await?;
    Ok(published.values[..outputs]
        .iter_mut()
        .map(|value| value.take().expect("published ILU output is present"))
        .collect())
}

fn class(source: &SparseTensor) -> BuiltinResult<ExactHostSparseClass> {
    if source.as_complex_f32_slice().is_some() {
        Ok(ExactHostSparseClass::Complex(NumericDType::F32))
    } else if source.as_complex_f64_slice().is_some() {
        Ok(ExactHostSparseClass::Complex(NumericDType::F64))
    } else {
        source
            .numeric_dtype()
            .map(ExactHostSparseClass::Numeric)
            .ok_or_else(|| error(&ILU_ERROR_INVALID_INPUT, None))
    }
}
async fn publish<T: ManagedSparseScalar>(
    factor: &ManagedIncompleteLu<T>,
    projection: &crate::builtins::math::linalg::ops::sparse_factor::IncompleteLuCsc<T>,
    source: &SparseTensor,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    factor.check_owner(runtime).map_err(|e| {
        crate::context::host_allocation::host_allocation_runtime_error(
            Some(runtime),
            "RunMat:ilu:Runtime",
            "ilu: factor owner",
            e,
        )
    })?;
    publish_arrays(
        factor,
        projection.col_ptrs(),
        projection.rows(),
        projection.values(),
        class(source)?,
        factor.order(),
        runtime,
    )
    .await
}
async fn publish_arrays<T: ManagedSparseScalar>(
    factor: &ManagedIncompleteLu<T>,
    ptr: &[usize],
    rows: &[usize],
    values: &[T],
    class: ExactHostSparseClass,
    n: usize,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    let mut output = SparsePublication::new(runtime, class, n, n, values.len())
        .map_err(|e| publication_error(e, runtime))?;
    for (i, &v) in ptr.iter().enumerate() {
        output
            .pointer(i, v)
            .map_err(|e| publication_error(e, runtime))?;
    }
    for i in 0..values.len() {
        output
            .entry(i, rows[i], values[i].parts())
            .map_err(|e| publication_error(e, runtime))?;
        if i % 256 == 255 {
            output
                .checkpoint(runtime)
                .map_err(|e| publication_error(e, runtime))?;
        }
    }
    output
        .finish_with(
            runtime,
            || {
                factor
                    .check_owner(runtime)
                    .map_err(|e| factor_error(ManagedIncompleteLuError::Admission(e), runtime))
            },
            |e| publication_error(e, runtime),
        )
        .await
}
async fn publish_combined<T: ManagedSparseScalar>(
    factor: &ManagedIncompleteLu<T>,
    lower: &crate::builtins::math::linalg::ops::sparse_factor::IncompleteLuCsc<T>,
    upper: &crate::builtins::math::linalg::ops::sparse_factor::IncompleteLuCsc<T>,
    source: &SparseTensor,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    // L and U overlap only on the unit diagonal; W retains U's diagonal.
    let n = factor.order();
    let nnz = lower
        .values()
        .len()
        .checked_add(upper.values().len())
        .and_then(|v| v.checked_sub(n))
        .ok_or_else(|| error(&ILU_ERROR_RUNTIME, Some("ilu: combined output overflow")))?;
    let c = class(source)?;
    let mut out =
        SparsePublication::new(runtime, c, n, n, nnz).map_err(|e| publication_error(e, runtime))?;
    let mut written = 0;
    out.pointer(0, 0)
        .map_err(|e| publication_error(e, runtime))?;
    for col in 0..n {
        for at in upper.col_ptrs()[col]..upper.col_ptrs()[col + 1] {
            out.entry(written, upper.rows()[at], upper.values()[at].parts())
                .map_err(|e| publication_error(e, runtime))?;
            written += 1;
        }
        for at in lower.col_ptrs()[col]..lower.col_ptrs()[col + 1] {
            if lower.rows()[at] != col {
                out.entry(written, lower.rows()[at], lower.values()[at].parts())
                    .map_err(|e| publication_error(e, runtime))?;
                written += 1;
            }
        }
        out.pointer(col + 1, written)
            .map_err(|e| publication_error(e, runtime))?;
        out.checkpoint(runtime)
            .map_err(|e| publication_error(e, runtime))?;
    }
    out.finish_with(
        runtime,
        || {
            factor
                .check_owner(runtime)
                .map_err(|e| factor_error(ManagedIncompleteLuError::Admission(e), runtime))
        },
        |e| publication_error(e, runtime),
    )
    .await
}
async fn publish_permutation<T: ManagedSparseScalar>(
    factor: &ManagedIncompleteLu<T>,
    transpose: bool,
    runtime: &RuntimeContext,
) -> BuiltinResult<Value> {
    let n = factor.order();
    let c = ExactHostSparseClass::Numeric(NumericDType::F64);
    let mut out =
        SparsePublication::new(runtime, c, n, n, n).map_err(|e| publication_error(e, runtime))?;
    for col in 0..=n {
        out.pointer(col, col)
            .map_err(|e| publication_error(e, runtime))?;
    }
    if transpose {
        for original in 0..n {
            out.entry(original, factor.inverse_permutation()[original], (1.0, 0.0))
                .map_err(|e| publication_error(e, runtime))?;
        }
    } else {
        for (live, &original) in factor.column_permutation().iter().enumerate() {
            out.entry(live, original, (1.0, 0.0))
                .map_err(|e| publication_error(e, runtime))?;
        }
    }
    out.finish_with(
        runtime,
        || {
            factor
                .check_owner(runtime)
                .map_err(|e| factor_error(ManagedIncompleteLuError::Admission(e), runtime))
        },
        |e| publication_error(e, runtime),
    )
    .await
}
fn publication_error(e: SparsePublicationError, runtime: &RuntimeContext) -> RuntimeError {
    match e {
        SparsePublicationError::Admission(source) => {
            crate::context::host_allocation::host_allocation_runtime_error(
                Some(runtime),
                "RunMat:ilu:Runtime",
                "ilu: sparse output allocation",
                source,
            )
        }
        SparsePublicationError::Execution(source) => {
            source.into_runtime_error_in_context(Some(runtime))
        }
        other => error(&ILU_ERROR_RUNTIME, Some(&other.to_string())),
    }
}
fn factor_error(e: ManagedIncompleteLuError, runtime: &RuntimeContext) -> RuntimeError {
    match e {
        ManagedIncompleteLuError::Admission(s) => {
            crate::context::host_allocation::host_allocation_runtime_error(
                Some(runtime),
                "RunMat:ilu:Runtime",
                "ilu: sparse factor allocation",
                s,
            )
        }
        ManagedIncompleteLuError::Execution(s) => s.into_runtime_error_in_context(Some(runtime)),
        ManagedIncompleteLuError::Singular { .. }
        | ManagedIncompleteLuError::NumericalBreakdown => {
            error(&ILU_ERROR_NUMERICAL, Some(&e.to_string()))
        }
        ManagedIncompleteLuError::NotSquare
        | ManagedIncompleteLuError::UnsupportedClass
        | ManagedIncompleteLuError::NonFiniteInput
        | ManagedIncompleteLuError::InvalidTopology(_) => {
            error(&ILU_ERROR_INVALID_INPUT, Some(&e.to_string()))
        }
    }
}
