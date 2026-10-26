//! Original physical output and independently checked NNLS primal/dual conditions.
use super::*;
use runmat_value::SparseTensor;
pub(super) async fn residual(
    matrix: &SparseTensor,
    x: &[f64],
    d: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    if x.len() != matrix.cols || d.len() != matrix.rows {
        return Err(numerical("invalid numerical geometry"));
    }
    let mut residual = runmat_value::exact_capacity_vec(d.len()).map_err(allocation)?;
    for chunk in d.chunks(1024) {
        residual.extend_from_slice(chunk);
        cooperation.completed(chunk.len() as u64).await?;
    }
    let values = matrix.as_f64_slice().expect("normalized real data");
    for (col, &x) in x.iter().enumerate() {
        for k in matrix.col_ptrs[col]..matrix.col_ptrs[col + 1] {
            residual[matrix.row_indices[k]] -= values[k] * x;
            if k % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        cooperation.completed(1).await?;
    }
    if residual.iter().any(|v| !v.is_finite()) {
        return Err(numerical(
            "original residual exceeds finite arithmetic range",
        ));
    }
    Ok(residual)
}
pub(super) async fn dual(
    matrix: &SparseTensor,
    residual: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let values = matrix.as_f64_slice().expect("normalized real data");
    let mut dual = runmat_value::exact_capacity_vec(matrix.cols).map_err(allocation)?;
    for col in 0..matrix.cols {
        let mut sum = 0.0_f64;
        let mut correction = 0.0_f64;
        for k in matrix.col_ptrs[col]..matrix.col_ptrs[col + 1] {
            let term = values[k] * residual[matrix.row_indices[k]];
            let next = sum + term;
            if !next.is_finite() {
                return Err(numerical(
                    "dual accumulation exceeds finite arithmetic range",
                ));
            }
            correction += if sum.abs() >= term.abs() {
                (sum - next) + term
            } else {
                (term - next) + sum
            };
            sum = next;
            if k % 1024 == 1023 {
                cooperation.completed(1024).await?;
            }
        }
        let value = sum + correction;
        if !value.is_finite() {
            return Err(numerical("nonfinite dual accumulation"));
        }
        dual.push(value);
        cooperation.completed(1).await?;
    }
    Ok(dual)
}
pub(super) async fn finalize(
    model: &model::Model,
    v: Vec<f64>,
    flag: i32,
    threshold: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<model::Outcome> {
    let mut x = v;
    for chunk in x.chunks_mut(1024) {
        for value in chunk.iter_mut() {
            let working = *value;
            *value = if model.scale == 0.0 {
                0.0
            } else {
                working / model.scale
            };
            if working > 0.0 && *value == 0.0 {
                return Err(numerical(
                    "physical solution underflows its working coordinate",
                ));
            }
            if !value.is_finite() || *value < 0.0 {
                return Err(numerical("physical solution is not finite and nonnegative"));
            }
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    // Recompute from original C and physical x, not the scaled working iterate.
    let residual = residual(&model.c, &x, &model.d, cooperation).await?;
    let scaled_dual = dual(&model.scaled, &residual, cooperation).await?;
    if flag == 1 {
        for (&point, &dual) in x.iter().zip(&scaled_dual) {
            if dual > threshold || (point > 0.0 && dual.abs() > threshold) {
                return Err(numerical(
                    "candidate failed original nonnegativity/stationarity checks",
                ));
            }
            cooperation.completed(1).await?;
        }
        let max_x = x.iter().copied().fold(0.0_f64, f64::max);
        let max_dual = scaled_dual.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if max_x > 0.0 && max_dual > 0.0 {
            let mut weighted = 0.0;
            let mut weights = 0.0;
            for (&x, &dual) in x.iter().zip(&scaled_dual) {
                let weight = x / max_x;
                weights += weight;
                weighted += weight * (dual.abs() / max_dual);
                cooperation.completed(1).await?;
            }
            // Equivalent to sum|x_i*lambda_i| <= TolX*sum x_i,
            // without overflowing or underflowing physical products.
            if weighted / weights > threshold / max_dual {
                return Err(numerical("candidate failed original complementarity check"));
            }
        }
    }
    let mut lambda = scaled_dual;
    for chunk in lambda.chunks_mut(1024) {
        for dual in chunk.iter_mut() {
            *dual *= model.scale;
            if !dual.is_finite() {
                return Err(numerical("physical dual exceeds finite arithmetic range"));
            }
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    let mut norm = 0.0_f64;
    for chunk in residual.chunks(1024) {
        for value in chunk {
            norm = norm.hypot(*value);
        }
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(model::Outcome {
        x,
        residual,
        resnorm: norm * norm,
        lambda,
        flag,
        message: if flag == 1 {
            "Nonnegative solution satisfies original residual and primal/dual checks."
        } else {
            "Iteration limit reached; returning the actual feasible iterate."
        },
    })
}
