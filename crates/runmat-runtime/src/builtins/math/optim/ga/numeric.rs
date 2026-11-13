//! Exact structural indices and original-context numeric materialization.
use super::{common, invalid, ScalarCooperation, IDENTITY, NAME, POLICY};
use crate::BuiltinResult;
use common::constrained_numeric as numeric;
use runmat_value::Value;

/// GA's public linear-row feasibility rule is distinct from its nonlinear
/// tolerance, including the documented nonnegative zero setting.
pub(in crate::builtins::math::optim) fn linear_tolerance(nonlinear: f64) -> f64 {
    nonlinear.max(f64::EPSILON.sqrt())
}

pub(super) const INPUT_POLICY: numeric::Policy = numeric::Policy {
    owner: IDENTITY,
    numeric: Some(&runmat_builtins::GA_NUMERIC_EXTENSION),
    single: Some(&runmat_builtins::GA_SINGLE_EXTENSION),
    invalid: Some(&runmat_builtins::GA_ERROR_INVALID_ARGUMENT),
    resident: None,
    sparse: None,
};

pub(super) fn preflight(
    value: &Value,
    role: &str,
    geometry: numeric::Geometry,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    if numeric::preflight(value, role, geometry, &INPUT_POLICY, runtime)? {
        Ok(())
    } else {
        Err(invalid(format!("{role} must be real numeric data")))
    }
}

pub(super) async fn variables(
    value: Value,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<usize> {
    let Some(value) = numeric::prepare(
        value,
        "nvars",
        numeric::Geometry::Scalar,
        &INPUT_POLICY,
        cooperation,
    )
    .await?
    else {
        return Err(invalid("nvars must be a positive integer-valued scalar"));
    };
    // The shared decoder preserves native integer scalar storage and rejects
    // an unrepresentable structural count without a source-driven rank pad.
    let count = common::option_usize_value(NAME, Some(&value), "nvars", 0)?;
    if count == 0 {
        return Err(invalid("nvars must be positive"));
    }
    Ok(count)
}

fn admitted_index(index: Option<usize>, n: usize) -> BuiltinResult<usize> {
    let index = index
        .filter(|&index| index > 0 && index <= n)
        .ok_or_else(|| {
            invalid("intcon must contain positive exact indices no greater than nvars")
        })?;
    Ok(index - 1)
}

pub(super) async fn indices(
    value: Value,
    n: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<usize>> {
    let Some(value) = numeric::prepare(
        value,
        "intcon",
        numeric::Geometry::Vector,
        &INPUT_POLICY,
        cooperation,
    )
    .await?
    else {
        return Err(invalid("intcon must be a real numeric index vector"));
    };
    let mut result;
    if let Value::Int(integer) = &value {
        result = POLICY.capacity(1, cooperation.runtime())?;
        result.push(admitted_index(integer.try_to_usize(), n)?);
    } else {
        let storage = match &value {
            Value::Tensor(tensor) => tensor.integer_storage(),
            Value::SparseTensor(sparse) if sparse.integer_storage().is_some() => {
                let elements = sparse
                    .rows
                    .checked_mul(sparse.cols)
                    .ok_or_else(|| invalid("intcon sparse shape overflow"))?;
                if sparse.nnz() != elements {
                    return Err(invalid("intcon cannot contain implicit zero indices"));
                }
                let mut validation = runmat_value::SparseCscValidation::new(
                    sparse.rows,
                    sparse.cols,
                    sparse.col_ptrs.clone(),
                    sparse.row_indices.clone(),
                    sparse.nnz(),
                )
                .map_err(invalid)?;
                loop {
                    let done = validation
                        .advance(std::num::NonZeroUsize::new(1024).expect("nonzero quantum"))
                        .map_err(invalid)?;
                    cooperation.completed(1024).await?;
                    if done {
                        break;
                    }
                }
                sparse.integer_storage()
            }
            _ => None,
        };
        if let Some(storage) = storage {
            result = POLICY.capacity(storage.len(), cooperation.runtime())?;
            for index in 0..storage.len() {
                result.push(admitted_index(
                    storage
                        .value_at(index)
                        .and_then(|value| value.try_to_usize()),
                    n,
                )?);
                cooperation.completed(1).await?;
            }
        } else {
            let numeric::Numeric::Real { values, .. } =
                numeric::materialize(value, "intcon", &INPUT_POLICY, cooperation).await?
            else {
                return Err(invalid("intcon must contain finite exact indices"));
            };
            result = POLICY.capacity(values.len(), cooperation.runtime())?;
            for value in values {
                let index =
                    common::option_usize_value(NAME, Some(&Value::Num(value)), "intcon", 0)?;
                result.push(admitted_index(Some(index), n)?);
                cooperation.completed(1).await?;
            }
        }
    }
    use crate::builtins::common::cooperative_sort::{SortProgress, StableSortWork};
    let mut sort = StableSortWork::new(&mut result, usize::cmp);
    loop {
        let progress = sort.advance().map_err(|source| {
            crate::build_runtime_error("ga: integer index ordering allocation failed")
                .with_builtin(NAME)
                .with_identifier("RunMat:ga:AllocationFailed")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .with_source(source.0)
                .build()
        })?;
        let (done, work) = match progress {
            SortProgress::Pending(work) => (false, work),
            SortProgress::Complete(work) => (true, work),
        };
        cooperation.completed(work).await?;
        if done {
            break;
        }
    }
    drop(sort);
    let mut written = 0;
    for read in 0..result.len() {
        if written == 0 || result[written - 1] != result[read] {
            result[written] = result[read];
            written += 1;
        }
        cooperation.completed(1).await?;
    }
    result.truncate(written);
    Ok(result)
}

pub(super) async fn pair(
    matrix: Value,
    rhs: Value,
    n: usize,
    role: &str,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(clarabel::algebra::CscMatrix<f64>, Vec<f64>)> {
    let Some(matrix) = numeric::prepare(
        matrix,
        role,
        numeric::Geometry::MatrixAny,
        &INPUT_POLICY,
        cooperation,
    )
    .await?
    else {
        return Err(invalid(format!("{role} must be real")));
    };
    let matrix = common::conic::matrix::parse(
        matrix,
        common::conic::matrix::IntegerPolicy::Exact,
        cooperation,
        |detail: &str| invalid(format!("{role} {detail}")),
    )
    .await?;
    let numeric::Numeric::Real { values: rhs, .. } = numeric::read(
        rhs,
        role,
        numeric::Geometry::Vector,
        &INPUT_POLICY,
        cooperation,
    )
    .await?
    else {
        return Err(invalid(format!("{role} right sides must be finite")));
    };
    let matrix = match matrix {
        None if rhs.is_empty() => clarabel::algebra::CscMatrix::zeros((0, n)),
        Some(matrix) if matrix.n == n && matrix.m == rhs.len() => matrix,
        _ => {
            return Err(invalid(format!(
                "{role} row counts must match and matrices must have nvars columns"
            )))
        }
    };
    Ok((matrix, rhs))
}

pub(super) async fn bound(
    value: Value,
    n: usize,
    default: f64,
    role: &str,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let Some(value) = numeric::prepare(
        value,
        role,
        numeric::Geometry::Vector,
        &INPUT_POLICY,
        cooperation,
    )
    .await?
    else {
        return Err(invalid(format!("{role} must be real")));
    };
    let numeric::Numeric::Real { values, .. } =
        numeric::materialize_raw(value, role, &INPUT_POLICY, cooperation).await?
    else {
        return Err(invalid(format!("{role} must be real")));
    };
    if values.len() > n || values.iter().any(|value| value.is_nan()) {
        return Err(invalid(format!(
            "{role} must have at most nvars entries and cannot contain NaN"
        )));
    }
    let mut bound = POLICY.capacity(n, cooperation.runtime())?;
    for value in values {
        bound.push(value);
        cooperation.completed(1).await?;
    }
    while bound.len() < n {
        bound.push(default);
        cooperation.completed(1).await?;
    }
    Ok(bound)
}
