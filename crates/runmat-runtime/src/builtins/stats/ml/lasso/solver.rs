use super::*;

pub(super) fn default_lambda_sequence(
    data: &PreparedData,
    options: &LassoOptions,
) -> BuiltinResult<Vec<f64>> {
    let lambda_max = (0..data.cols)
        .map(|col| {
            (0..data.rows)
                .map(|row| data.weights[row] * x_at(data, row, col) * data.y_work[row])
                .sum::<f64>()
                .abs()
        })
        .fold(0.0, f64::max)
        / options.alpha.max(EPS);
    if options.num_lambda == 1 {
        return Ok(vec![lambda_max]);
    }
    if lambda_max <= EPS {
        return Ok(vec![0.0]);
    }
    let ratio = options.lambda_ratio;
    if ratio <= EPS {
        let mut values = geometric_sequence(lambda_max, 1.0e-4, options.num_lambda);
        if let Some(last) = values.last_mut() {
            *last = 0.0;
        }
        values.sort_by(|a, b| a.total_cmp(b));
        return Ok(values);
    }
    let mut values = geometric_sequence(lambda_max, ratio, options.num_lambda);
    values.sort_by(|a, b| a.total_cmp(b));
    Ok(values)
}

pub(super) fn sorted_unique_lambdas(lambda: &[f64]) -> BuiltinResult<Vec<f64>> {
    super::super::regularized::sorted_unique_lambdas(lambda, EPS)
        .ok_or_else(|| invalid_argument("lasso: Lambda must not be empty"))
}

pub(super) fn fit_path(
    data: &PreparedData,
    lambdas_ascending: &[f64],
    options: &LassoOptions,
) -> BuiltinResult<FitPath> {
    let mut descending = lambdas_ascending.to_vec();
    descending.sort_by(|a, b| b.total_cmp(a));
    let mut beta_scaled = vec![0.0; data.cols];
    let mut entries = Vec::with_capacity(descending.len());
    for lambda in descending {
        let iterations = fit_single_lambda(data, &mut beta_scaled, lambda, options)?;
        let beta_original = unstandardize_coefficients(data, &beta_scaled);
        let intercept = if options.intercept {
            data.y_mean
                - beta_original
                    .iter()
                    .zip(data.x_means.iter())
                    .map(|(beta, mean)| beta * mean)
                    .sum::<f64>()
        } else {
            0.0
        };
        let mse = weighted_mse(data, &beta_original, intercept);
        let df = beta_original
            .iter()
            .filter(|value| value.abs() > 1.0e-8)
            .count() as f64;
        entries.push((lambda, beta_original, intercept, df, mse, iterations as f64));
    }
    entries.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(FitPath {
        lambdas: entries.iter().map(|entry| entry.0).collect(),
        coefficients: entries.iter().map(|entry| entry.1.clone()).collect(),
        intercepts: entries.iter().map(|entry| entry.2).collect(),
        df: entries.iter().map(|entry| entry.3).collect(),
        mse: entries.iter().map(|entry| entry.4).collect(),
        iterations: entries.iter().map(|entry| entry.5).collect(),
    })
}

pub(super) fn fit_single_lambda(
    data: &PreparedData,
    beta: &mut [f64],
    lambda: f64,
    options: &LassoOptions,
) -> BuiltinResult<usize> {
    let mut residual = data
        .y_work
        .iter()
        .enumerate()
        .map(|(row, y)| {
            let predicted = (0..data.cols)
                .map(|col| x_at(data, row, col) * beta[col])
                .sum::<f64>();
            y - predicted
        })
        .collect::<Vec<_>>();
    let mut column_norms = vec![0.0; data.cols];
    for (col, norm) in column_norms.iter_mut().enumerate() {
        *norm = (0..data.rows)
            .map(|row| data.weights[row] * x_at(data, row, col) * x_at(data, row, col))
            .sum::<f64>();
    }
    for iter in 1..=options.max_iter {
        let mut max_delta = 0.0_f64;
        let mut max_beta = 0.0_f64;
        for col in 0..data.cols {
            if column_norms[col] <= EPS {
                beta[col] = 0.0;
                continue;
            }
            let old = beta[col];
            let rho = (0..data.rows)
                .map(|row| {
                    data.weights[row]
                        * x_at(data, row, col)
                        * (residual[row] + x_at(data, row, col) * old)
                })
                .sum::<f64>();
            let denom = column_norms[col] + lambda * (1.0 - options.alpha);
            if denom <= EPS || !denom.is_finite() {
                return Err(convergence_error(
                    "lasso: coordinate descent encountered a singular predictor column",
                ));
            }
            let new_beta = soft_threshold(rho, lambda * options.alpha) / denom;
            let delta = new_beta - old;
            if delta != 0.0 {
                for (row, value) in residual.iter_mut().enumerate() {
                    *value -= x_at(data, row, col) * delta;
                }
                beta[col] = new_beta;
            }
            max_delta = max_delta.max(delta.abs());
            max_beta = max_beta.max(new_beta.abs());
        }
        if max_delta <= options.rel_tol * max_beta.max(1.0) {
            return Ok(iter);
        }
    }
    Ok(options.max_iter)
}

pub(super) fn x_at(data: &PreparedData, row: usize, col: usize) -> f64 {
    data.x_work[row + col * data.rows]
}

pub(super) fn unstandardize_coefficients(data: &PreparedData, beta_scaled: &[f64]) -> Vec<f64> {
    beta_scaled
        .iter()
        .zip(data.x_scales.iter())
        .map(|(beta, scale)| beta / scale)
        .collect()
}

pub(super) fn weighted_mse(data: &PreparedData, beta: &[f64], intercept: f64) -> f64 {
    (0..data.rows)
        .map(|row| {
            let predicted = intercept
                + (0..data.cols)
                    .map(|col| data.x_original[row + col * data.rows] * beta[col])
                    .sum::<f64>();
            let err = data.y_original[row] - predicted;
            data.weights[row] * err * err
        })
        .sum()
}

pub(super) fn apply_df_limit(mut path: FitPath, df_max: Option<usize>) -> BuiltinResult<FitPath> {
    let Some(max_df) = df_max else {
        return Ok(path);
    };
    let keep = path
        .df
        .iter()
        .enumerate()
        .filter_map(|(idx, df)| (*df as usize <= max_df).then_some(idx))
        .collect::<Vec<_>>();
    if keep.is_empty() {
        return Err(invalid_argument(
            "lasso: DFmax excludes every fitted Lambda value",
        ));
    }
    path.lambdas = keep.iter().map(|idx| path.lambdas[*idx]).collect();
    path.coefficients = keep
        .iter()
        .map(|idx| path.coefficients[*idx].clone())
        .collect();
    path.intercepts = keep.iter().map(|idx| path.intercepts[*idx]).collect();
    path.df = keep.iter().map(|idx| path.df[*idx]).collect();
    path.mse = keep.iter().map(|idx| path.mse[*idx]).collect();
    path.iterations = keep.iter().map(|idx| path.iterations[*idx]).collect();
    Ok(path)
}
