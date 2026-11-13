//! Dense eig consumers borrow coefficients and fill the canonical EVD owner.
use super::*;
use crate::builtins::math::linalg::ops::dense_evd::{
    DenseEvdError, Eigenpairs, PreparedEvd, Problem, Vectors,
};
use crate::context::RuntimeContext;

pub(in crate::builtins::math::linalg::factor) fn error(
    runtime: Option<&RuntimeContext>,
    source: DenseEvdError,
) -> RuntimeError {
    match source {
        DenseEvdError::Admission(source) => with_eig_context(
            crate::context::host_allocation::host_allocation_runtime_error(
                runtime,
                "RunMat:eig:Workspace",
                "eig: dense EVD allocation failed",
                source,
            ),
        ),
        DenseEvdError::Execution(source) => source.into_runtime_error_in_context(runtime),
        #[cfg(all(feature = "blas-lapack", not(target_arch = "wasm32")))]
        DenseEvdError::LapackFailure(info) => {
            let message = if info < 0 {
                format!("eig: LAPACK ZGGEV rejected argument {}", -i64::from(info))
            } else {
                format!("eig: LAPACK ZGGEV failed to converge with info = {info}")
            };
            let mut error = eig_internal_error(message);
            error.source = Some(Box::new(DenseEvdError::LapackFailure(info)));
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        }
        source => {
            let mut error = eig_internal_error(format!("eig: {source}"));
            error.source = Some(Box::new(source));
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
            error
        }
    }
}

/// Validation is separate from allocation so the original A/B error order
/// does not depend on how much memory the execution policy admits.
pub(in crate::builtins::math::linalg::factor) fn dimensions(
    value: &Value,
    sparse: bool,
) -> BuiltinResult<(usize, usize)> {
    let geometry = match value {
        Value::Tensor(value) => {
            if value.shape().len() > 2 {
                return Err(eig_invalid_input("eig: input must be 2-D"));
            }
            (value.rows(), value.cols())
        }
        Value::ComplexTensor(value) => {
            if value.shape().len() > 2 {
                return Err(eig_invalid_input("eig: input must be 2-D"));
            }
            (value.rows(), value.cols())
        }
        Value::LogicalArray(value) => {
            if value.shape.len() > 2 {
                return Err(eig_invalid_input("eig: input must be 2-D"));
            }
            match value.shape.as_slice() {
                [] => (1, 1),
                [n] => (*n, 1),
                [r, c] => (*r, *c),
                _ => unreachable!(),
            }
        }
        Value::SparseTensor(value) if sparse => (value.rows, value.cols),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => (1, 1),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
            return Err(eig_invalid_input(
                "eig: input must be numeric or logical; convert character data with double() first",
            ))
        }
        other => {
            return Err(eig_invalid_input(format!(
                "eig: unsupported input type {other:?}; expected numeric or logical values"
            )))
        }
    };
    geometry
        .0
        .checked_mul(geometry.1)
        .ok_or_else(|| eig_invalid_input("eig: matrix dimensions exceed addressable storage"))?;
    Ok(geometry)
}

pub(in crate::builtins::math::linalg::factor) fn element(value: &Value, index: usize) -> Complex64 {
    match value {
        Value::Tensor(value) => Complex64::new(
            value
                .numeric_value_at(index)
                .expect("validated dense coefficient")
                .materialize_f64(),
            0.0,
        ),
        Value::ComplexTensor(value) => {
            let (re, im) = value
                .numeric_value_at(index)
                .expect("validated complex coefficient");
            Complex64::new(re.materialize_f64(), im.materialize_f64())
        }
        Value::LogicalArray(value) => {
            Complex64::new(if value.data[index] != 0 { 1.0 } else { 0.0 }, 0.0)
        }
        Value::Num(value) => Complex64::new(*value, 0.0),
        Value::Int(value) => Complex64::new(value.to_f64(), 0.0),
        Value::Bool(value) => Complex64::new(if *value { 1.0 } else { 0.0 }, 0.0),
        Value::Complex(re, im) => Complex64::new(*re, *im),
        _ => unreachable!("sparse coefficients use the CSC fill, all other classes validated"),
    }
}

pub(in crate::builtins::math::linalg::factor) fn coefficient(
    value: &Value,
    index: usize,
) -> Complex64 {
    if let Value::SparseTensor(value) = value {
        let row = index % value.rows;
        let column = index / value.rows;
        let start = value.col_ptrs[column];
        let end = value.col_ptrs[column + 1];
        return value.row_indices[start..end]
            .binary_search(&row)
            .ok()
            .map_or(Complex64::new(0., 0.), |offset| {
                let slot = start + offset;
                if let Some(runmat_value::ComplexElement(re, im)) = value.complex_value_at(slot) {
                    Complex64::new(re, im)
                } else {
                    Complex64::new(
                        value
                            .numeric_value_at(slot)
                            .expect("validated CSC coefficient")
                            .materialize_f64(),
                        0.,
                    )
                }
            });
    }
    element(value, index)
}

pub(in crate::builtins::math::linalg::factor) async fn fill(
    value: &Value,
    prepared: &mut PreparedEvd,
    mass: bool,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    let mut budget = runtime.map(RuntimeContext::cpu_cooperation);
    prepared
        .check_owner(runtime)
        .map_err(|e| error(runtime, e))?;
    if let Value::SparseTensor(value) = value {
        let mut completed = 0u64;
        for column in 0..value.cols {
            for slot in value.col_ptrs[column]..value.col_ptrs[column + 1] {
                let entry = if let Some(runmat_value::ComplexElement(re, im)) =
                    value.complex_value_at(slot)
                {
                    Complex64::new(re, im)
                } else {
                    Complex64::new(
                        value
                            .numeric_value_at(slot)
                            .expect("validated CSC coefficient")
                            .materialize_f64(),
                        0.,
                    )
                };
                let output = if mass {
                    prepared.mass_mut().expect("generalized workspace")
                } else {
                    prepared.matrix_mut()
                };
                output[value.row_indices[slot] + column * value.rows] = entry;
                completed += 1;
                if completed == 4096 {
                    if let Some(budget) = &mut budget {
                        budget
                            .checkpoint(completed)
                            .await
                            .map_err(|e| e.into_runtime_error_in_context(runtime))?;
                    }
                    prepared
                        .check_owner(runtime)
                        .map_err(|e| error(runtime, e))?;
                    completed = 0;
                }
            }
        }
        if let Some(budget) = &mut budget {
            budget
                .checkpoint(completed)
                .await
                .map_err(|e| e.into_runtime_error_in_context(runtime))?;
        }
    } else {
        let len = prepared.matrix_mut().len();
        for start in (0..len).step_by(4096) {
            prepared
                .check_owner(runtime)
                .map_err(|e| error(runtime, e))?;
            let end = (start + 4096).min(len);
            {
                let output = if mass {
                    prepared.mass_mut().expect("generalized workspace")
                } else {
                    prepared.matrix_mut()
                };
                for (offset, slot) in output[start..end].iter_mut().enumerate() {
                    *slot = element(value, start + offset);
                }
            }
            if let Some(budget) = &mut budget {
                budget
                    .checkpoint((end - start) as u64)
                    .await
                    .map_err(|e| e.into_runtime_error_in_context(runtime))?;
            }
            prepared
                .check_owner(runtime)
                .map_err(|e| error(runtime, e))?;
        }
    }
    prepared
        .check_owner(runtime)
        .map_err(|e| error(runtime, e))?;
    Ok(())
}

pub(in crate::builtins::math::linalg::factor) async fn compute(
    runtime: Option<&RuntimeContext>,
    a: &Value,
    b: Option<&Value>,
    require_left: bool,
    sparse: bool,
) -> BuiltinResult<Eigenpairs> {
    let (rows, cols) = dimensions(a, sparse)?;
    let mass = b.map(|value| dimensions(value, sparse)).transpose()?;
    if let Some((br, bc)) = mass {
        if rows != cols {
            return Err(eig_invalid_input("eig: A matrix must be square"));
        }
        if br != bc {
            return Err(eig_invalid_input("eig: B matrix must be square"));
        }
        if (rows, cols) != (br, bc) {
            return Err(eig_invalid_input(
                "eig: A and B must be the same size for generalized eigenvalue decomposition",
            ));
        }
    } else if rows != cols {
        return Err(eig_error(&EIG_ERROR_INVALID_INPUT));
    }
    let mut prepared = PreparedEvd::new(
        rows,
        if b.is_some() {
            Problem::Generalized
        } else {
            Problem::Standard
        },
        if require_left {
            Vectors::LeftAndRight
        } else {
            Vectors::Right
        },
        runtime,
    )
    .await
    .map_err(|e| error(runtime, e))?;
    fill(a, &mut prepared, false, runtime).await?;
    if let Some(b) = b {
        fill(b, &mut prepared, true, runtime).await?;
    }
    prepared
        .compute(runtime)
        .await
        .map_err(|e| error(runtime, e))
}

pub(super) async fn publish(
    runtime: Option<&RuntimeContext>,
    pairs: &Eigenpairs,
    dtype: NumericDType,
    options: EigOptions,
) -> BuiltinResult<EigEval> {
    use crate::builtins::math::linalg::ops::dense_evd::publication::matrix;
    pairs.check_owner(runtime).map_err(|e| error(runtime, e))?;
    let n = pairs.order();
    let (rows, cols) = if n == 0 { (0, 0) } else { (n, 1) };
    let mut eigenvalues = matrix(runtime, dtype, rows, cols, |i| pairs.values()[i])
        .await
        .map_err(with_eig_context)?;
    if dtype == NumericDType::F64 {
        eigenvalues = match eigenvalues {
            Value::Tensor(value) => tensor::tensor_into_value(value),
            other => other,
        };
    }
    pairs.check_owner(runtime).map_err(|e| error(runtime, e))?;
    let diagonal_matrix = matrix(runtime, dtype, n, n, |i| {
        if i % n == i / n {
            pairs.values()[i / n]
        } else {
            Complex64::new(0., 0.)
        }
    })
    .await
    .map_err(with_eig_context)?;
    pairs.check_owner(runtime).map_err(|e| error(runtime, e))?;
    let right = matrix(runtime, dtype, n, n, |i| pairs.right()[i])
        .await
        .map_err(with_eig_context)?;
    pairs.check_owner(runtime).map_err(|e| error(runtime, e))?;
    let left = if let Some(values) = pairs.left() {
        Some(
            matrix(runtime, dtype, n, n, |i| values[i])
                .await
                .map_err(with_eig_context)?,
        )
    } else {
        None
    };
    pairs.check_owner(runtime).map_err(|e| error(runtime, e))?;
    Ok(EigEval {
        diagonal_output: if options.vector_output {
            eigenvalues.clone()
        } else {
            diagonal_matrix.clone()
        },
        eigenvalues,
        diagonal_matrix,
        right,
        left,
    })
}
