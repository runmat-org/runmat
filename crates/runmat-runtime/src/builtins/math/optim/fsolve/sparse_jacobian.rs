//! Canonical sparse CSC is validated and converted once at the dense LM boundary.
use super::*;
use runmat_value::SparseTensor;

pub(super) async fn row_major(
    sparse: &SparseTensor,
    rows: usize,
    cols: usize,
    cooperation: &mut crate::builtins::math::optim::common::ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let invalid = |detail: &str| fsolve_error_with_detail(&FSOLVE_ERROR_INVALID_INPUT, detail);
    if sparse.rows != rows || sparse.cols != cols || sparse.is_complex() {
        return Err(invalid("Jacobian must be a real m-by-n matrix"));
    }
    let count =
        crate::builtins::math::optim::least_squares::checked_least_squares_work(NAME, rows, cols)?;
    if sparse.integer_storage().is_some() || sparse.is_logical() {
        crate::compatibility::ensure_builtin_extension_enabled_in_optional_context(
            cooperation.runtime(),
            &FSOLVE_CALLBACK_NUMERIC_EXTENSION,
            NAME,
        )?;
    }
    sparse
        .cols
        .checked_add(1)
        .ok_or_else(|| invalid("Jacobian CSC column count overflow"))?;
    let mut validation = runmat_value::SparseCscValidation::new(
        rows,
        cols,
        sparse.col_ptrs.clone(),
        sparse.row_indices.clone(),
        sparse.nnz(),
    )
    .map_err(|message| invalid(&message))?;
    let work = std::num::NonZeroUsize::new(1024).expect("nonzero validation work");
    loop {
        let done = validation
            .advance(work)
            .map_err(|message| invalid(&message))?;
        cooperation.completed(1024).await?;
        if done {
            break;
        }
    }
    let value_at = |index| -> BuiltinResult<f64> {
        let value = if sparse.is_logical() {
            1.0
        } else {
            let scalar = sparse
                .numeric_host_buffer()
                .and_then(|values| values.value_at(index))
                .ok_or_else(|| invalid("Jacobian sparse numeric storage is invalid"))?;
            if let Some(integer) = scalar.into_int_value() {
                integer.to_exact_f64().ok_or_else(|| {
                    invalid("integer Jacobian must be exactly representable as double")
                })?
            } else {
                scalar.materialize_f64()
            }
        };
        if !value.is_finite() {
            return Err(invalid("Jacobian must contain finite values"));
        }
        Ok(value)
    };
    for first in (0..sparse.nnz()).step_by(1024) {
        let end = first.saturating_add(1024).min(sparse.nnz());
        for index in first..end {
            value_at(index)?;
        }
        cooperation.completed((end - first) as u64).await?;
    }
    let mut result = runmat_value::exact_capacity_vec(count).map_err(|source| {
        crate::build_runtime_error("Jacobian allocation failed")
            .with_source(source)
            .build()
    })?;
    while result.len() < count {
        let end = result.len().saturating_add(1024).min(count);
        let completed = end - result.len();
        result.resize(end, 0.0);
        cooperation.completed(completed as u64).await?;
    }
    for col in 0..cols {
        let start = sparse.col_ptrs[col];
        let end = sparse.col_ptrs[col + 1];
        for first in (start..end).step_by(1024) {
            let end = first.saturating_add(1024).min(end);
            for index in first..end {
                result[sparse.row_indices[index] * cols + col] = value_at(index)?;
            }
            cooperation.completed((end - first) as u64).await?;
        }
    }
    Ok(result)
}
