use super::*;

pub(super) fn lscov_compute(
    a: NumericMatrix,
    b: NumericMatrix,
    parsed: ParsedArgs,
    requested_outputs: usize,
) -> BuiltinResult<Vec<Value>> {
    let rows = matrix_rows(&a);
    let cols = matrix_cols(&a);
    if rows == 0 || cols == 0 {
        return Err(invalid("lscov: A must be a nonempty 2-D matrix"));
    }
    ensure_2d(&a, "A")?;
    ensure_2d(&b, "B")?;
    ensure_budget(rows, cols, "design matrix")?;
    let rhs_cols = rhs_columns(&b, rows)?;
    ensure_work_budget(cols)?;
    ensure_output_budget(cols, rhs_cols, requested_outputs)?;
    if requested_outputs == 4 && rhs_cols != 1 {
        return Err(invalid(
            "lscov: fourth output S is supported only when B is a vector",
        ));
    }
    match (a, b) {
        (NumericMatrix::Real(a), NumericMatrix::Real(b)) => {
            lscov_real(a, b, parsed, rhs_cols, requested_outputs)
        }
        (a, b) => lscov_complex(a, b, parsed, rhs_cols, requested_outputs),
    }
}

pub(super) fn lscov_real(
    a: Tensor,
    b: Tensor,
    parsed: ParsedArgs,
    rhs_cols: usize,
    requested_outputs: usize,
) -> BuiltinResult<Vec<Value>> {
    let (rows, cols) = lscov_dense_dimensions(&a);
    let a_values = tensor::tensor_values_f64_cow(&a);
    let a_mat = DMatrix::from_column_slice(rows, cols, a_values.as_ref());
    let b_mat = real_rhs_matrix(&b, rows, rhs_cols)?;
    let transformed = transform_real_problem(&a_mat, &b_mat, &parsed.weighting, parsed.algorithm)?;
    let solve = solve_real_least_squares(&transformed.a, &transformed.b, rows, cols)?;
    let transformed_residual = &transformed.b - &transformed.a * &solve.x;
    let mse = mse_real(&transformed_residual, rows, cols, solve.rank);

    let mut outputs = Vec::with_capacity(requested_outputs.max(1));
    outputs.push(real_tensor_value(
        solve.x.as_slice().to_vec(),
        vec![cols, rhs_cols],
        "x",
    )?);
    if requested_outputs == 1 {
        return Ok(outputs);
    }

    outputs.push(real_tensor_value(
        stdx_real(&solve.covariance_base, &mse, cols, rhs_cols),
        vec![cols, rhs_cols],
        "stdx",
    )?);
    if requested_outputs == 2 {
        return Ok(outputs);
    }

    outputs.push(real_tensor_value(mse.clone(), vec![1, rhs_cols], "mse")?);
    if requested_outputs == 3 {
        return Ok(outputs);
    }

    outputs.push(real_tensor_value(
        scaled_covariance_real(&solve.covariance_base, mse[0]),
        vec![cols, cols],
        "S",
    )?);
    Ok(outputs)
}

pub(super) fn lscov_complex(
    a: NumericMatrix,
    b: NumericMatrix,
    parsed: ParsedArgs,
    rhs_cols: usize,
    requested_outputs: usize,
) -> BuiltinResult<Vec<Value>> {
    let rows = matrix_rows(&a);
    let cols = matrix_cols(&a);
    let a_mat = complex_matrix(&a)?;
    let b_mat = complex_rhs_matrix(&b, rows, rhs_cols)?;
    let transformed =
        transform_complex_problem(&a_mat, &b_mat, &parsed.weighting, parsed.algorithm)?;
    let solve = solve_complex_least_squares(&transformed.a, &transformed.b, rows, cols)?;
    let transformed_residual = &transformed.b - &transformed.a * &solve.x;
    let mse = mse_complex(&transformed_residual, rows, cols, solve.rank);

    let mut outputs = Vec::with_capacity(requested_outputs.max(1));
    outputs.push(complex_matrix_value(&solve.x, cols, rhs_cols, "x")?);
    if requested_outputs == 1 {
        return Ok(outputs);
    }

    outputs.push(real_tensor_value(
        stdx_complex(&solve.covariance_base, &mse, cols, rhs_cols),
        vec![cols, rhs_cols],
        "stdx",
    )?);
    if requested_outputs == 2 {
        return Ok(outputs);
    }

    outputs.push(real_tensor_value(mse.clone(), vec![1, rhs_cols], "mse")?);
    if requested_outputs == 3 {
        return Ok(outputs);
    }

    outputs.push(complex_matrix_value(
        &(solve.covariance_base * Complex64::new(mse[0], 0.0)),
        cols,
        cols,
        "S",
    )?);
    Ok(outputs)
}

pub(super) fn mse_real(residual: &DMatrix<f64>, rows: usize, cols: usize, rank: usize) -> Vec<f64> {
    let rhs_cols = residual.ncols();
    if rows < cols {
        return vec![0.0; rhs_cols];
    }
    let dfe = rows as f64 - rank as f64;
    if dfe <= 0.0 {
        return vec![f64::NAN; rhs_cols];
    }
    (0..rhs_cols)
        .map(|col| residual.column(col).dot(&residual.column(col)) / dfe)
        .collect()
}

pub(super) fn mse_complex(
    residual: &DMatrix<Complex64>,
    rows: usize,
    cols: usize,
    rank: usize,
) -> Vec<f64> {
    let rhs_cols = residual.ncols();
    if rows < cols {
        return vec![0.0; rhs_cols];
    }
    let dfe = rows as f64 - rank as f64;
    if dfe <= 0.0 {
        return vec![f64::NAN; rhs_cols];
    }
    (0..rhs_cols)
        .map(|col| {
            residual
                .column(col)
                .iter()
                .map(|value| value.norm_sqr())
                .sum::<f64>()
                / dfe
        })
        .collect()
}

pub(super) fn stdx_real(
    covariance_base: &DMatrix<f64>,
    mse: &[f64],
    cols: usize,
    rhs_cols: usize,
) -> Vec<f64> {
    let mut out = vec![f64::NAN; cols * rhs_cols];
    for rhs_col in 0..rhs_cols {
        for col in 0..cols {
            let variance = covariance_base[(col, col)] * mse[rhs_col];
            out[col + rhs_col * cols] = if variance >= 0.0 {
                variance.sqrt()
            } else {
                f64::NAN
            };
        }
    }
    out
}

pub(super) fn stdx_complex(
    covariance_base: &DMatrix<Complex64>,
    mse: &[f64],
    cols: usize,
    rhs_cols: usize,
) -> Vec<f64> {
    let mut out = vec![f64::NAN; cols * rhs_cols];
    for rhs_col in 0..rhs_cols {
        for col in 0..cols {
            let variance = covariance_base[(col, col)].re * mse[rhs_col];
            out[col + rhs_col * cols] = if variance >= -EPS {
                variance.max(0.0).sqrt()
            } else {
                f64::NAN
            };
        }
    }
    out
}

pub(super) fn scaled_covariance_real(covariance_base: &DMatrix<f64>, mse: f64) -> Vec<f64> {
    covariance_base.iter().map(|value| value * mse).collect()
}
