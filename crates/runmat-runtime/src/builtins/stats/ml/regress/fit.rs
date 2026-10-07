use super::*;

pub(super) fn regress_compute(
    y: Tensor,
    x: Tensor,
    alpha: f64,
    requested_outputs: usize,
) -> BuiltinResult<Vec<Value>> {
    let y_values = vector_values(&y)?;
    if x.shape().len() > 2 {
        return Err(invalid("regress: X must be a 2-D numeric matrix"));
    }
    let (x_rows, x_cols) = regress_matrix_dimensions(x.shape());
    if x_rows != y_values.len() {
        return Err(invalid(
            "regress: length(y) must match the number of rows in X",
        ));
    }
    if x_cols == 0 {
        return Err(invalid("regress: X must contain at least one column"));
    }
    ensure_input_budget(x_rows, x_cols)?;
    let prepared = complete_rows(&x, &y_values)?;
    let rows = prepared.y.len();
    let cols = x_cols;
    if rows == 0 {
        return Err(invalid(
            "regress: at least one complete observation is required",
        ));
    }
    ensure_output_budget(rows, cols, requested_outputs)?;
    let active_columns = independent_columns(&prepared.design)?;
    ensure_diagnostic_budget(active_columns.len(), requested_outputs)?;

    let solve = solve_coefficients(&prepared, cols, active_columns, requested_outputs > 1)?;
    let residuals = &prepared.y - &solve.fitted;
    let sse = residuals.iter().map(|value| value * value).sum::<f64>();
    let y_mean = prepared.y.iter().sum::<f64>() / rows as f64;
    let sst = prepared
        .y
        .iter()
        .map(|value| {
            let diff = value - y_mean;
            diff * diff
        })
        .sum::<f64>();
    let dfe = rows as f64 - solve.rank as f64;
    let mse = if dfe > 0.0 { sse / dfe } else { f64::NAN };

    let mut outputs = Vec::with_capacity(requested_outputs.max(1));
    outputs.push(tensor_value(solve.beta.clone(), vec![cols, 1], "b")?);
    if requested_outputs == 1 {
        return Ok(outputs);
    }

    let inv_xtx_active = solve
        .inv_xtx_active
        .as_ref()
        .expect("diagnostic output requested inverse covariance");
    let tcrit = if dfe > 0.0 {
        student_t_inv(1.0 - alpha / 2.0, dfe)
    } else {
        f64::NAN
    };
    outputs.push(tensor_value(
        coefficient_intervals(
            &solve.beta,
            inv_xtx_active,
            &solve.active_columns,
            mse,
            tcrit,
        ),
        vec![cols, 2],
        "bint",
    )?);
    if requested_outputs == 2 {
        return Ok(outputs);
    }

    let residual_values = residuals.iter().copied().collect::<Vec<_>>();
    outputs.push(tensor_value(residual_values.clone(), vec![rows, 1], "r")?);
    if requested_outputs == 3 {
        return Ok(outputs);
    }

    outputs.push(tensor_value(
        residual_intervals(
            &prepared.design,
            inv_xtx_active,
            &solve.active_columns,
            &residual_values,
            sse,
            dfe,
            alpha,
        ),
        vec![rows, 2],
        "rint",
    )?);
    if requested_outputs == 4 {
        return Ok(outputs);
    }

    outputs.push(tensor_value(
        stats_values(sse, sst, mse, rows, solve.rank),
        vec![1, 4],
        "stats",
    )?);
    Ok(outputs)
}
