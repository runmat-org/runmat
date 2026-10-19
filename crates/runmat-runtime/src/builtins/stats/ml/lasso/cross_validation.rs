use super::*;

pub(super) fn cross_validate(
    x: &Tensor,
    y: &[f64],
    lambdas: &[f64],
    options: &LassoOptions,
    k: usize,
) -> BuiltinResult<CvInfo> {
    if k > y.len() {
        return Err(invalid_argument(
            "lasso: CV fold count cannot exceed the number of observations",
        ));
    }
    let mut fold_errors = vec![vec![0.0; lambdas.len()]; k];
    for fold in 0..k {
        let train_rows = (0..x.rows())
            .filter(|row| row % k != fold)
            .collect::<Vec<_>>();
        let test_rows = (0..x.rows())
            .filter(|row| row % k == fold)
            .collect::<Vec<_>>();
        let train_x = subset_rows_tensor(x, &train_rows)?;
        let train_y = train_rows.iter().map(|row| y[*row]).collect::<Vec<_>>();
        let mut fold_options = options.clone();
        fold_options.cv = CvSpec::Resubstitution;
        fold_options.lambda = Some(lambdas.to_vec());
        if let Some(weights) = &options.weights {
            fold_options.weights = Some(train_rows.iter().map(|row| weights[*row]).collect());
        }
        let prepared = prepare_data(&train_x, &train_y, &fold_options)?;
        let path = fit_path(&prepared, lambdas, &fold_options)?;
        for (lambda_idx, beta) in path.coefficients.iter().enumerate() {
            let intercept = path.intercepts[lambda_idx];
            fold_errors[fold][lambda_idx] = validation_mse(
                x,
                y,
                beta,
                intercept,
                &test_rows,
                options.weights.as_deref(),
            );
        }
    }
    let mut mse = vec![0.0; lambdas.len()];
    let mut se = vec![0.0; lambdas.len()];
    for lambda_idx in 0..lambdas.len() {
        let values = (0..k)
            .map(|fold| fold_errors[fold][lambda_idx])
            .collect::<Vec<_>>();
        let mean = values.iter().sum::<f64>() / k as f64;
        let variance = if k > 1 {
            values
                .iter()
                .map(|value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f64>()
                / (k - 1) as f64
        } else {
            0.0
        };
        mse[lambda_idx] = mean;
        se[lambda_idx] = variance.sqrt() / (k as f64).sqrt();
    }
    let index_min_mse = mse
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let threshold = mse[index_min_mse] + se[index_min_mse];
    let index_1se = mse
        .iter()
        .enumerate()
        .filter(|(_, value)| **value <= threshold)
        .max_by(|(left, _), (right, _)| lambdas[*left].total_cmp(&lambdas[*right]))
        .map(|(idx, _)| idx)
        .unwrap_or(index_min_mse);
    Ok(CvInfo {
        mse,
        se,
        lambda_min_mse: lambdas[index_min_mse],
        lambda_1se: lambdas[index_1se],
        index_min_mse: index_min_mse + 1,
        index_1se: index_1se + 1,
    })
}

pub(super) fn subset_rows_tensor(x: &Tensor, rows: &[usize]) -> BuiltinResult<Tensor> {
    let mut data = Vec::with_capacity(rows.len() * x.cols());
    for col in 0..x.cols() {
        for row in rows {
            data.push(
                x.get2(*row, col)
                    .map_err(|err| invalid_argument(format!("lasso: {err}")))?,
            );
        }
    }
    Tensor::new(data, vec![rows.len(), x.cols()])
        .map_err(|err| invalid_argument(format!("lasso: {err}")))
}

pub(super) fn validation_mse(
    x: &Tensor,
    y: &[f64],
    beta: &[f64],
    intercept: f64,
    rows: &[usize],
    weights: Option<&[f64]>,
) -> f64 {
    if rows.is_empty() {
        return f64::NAN;
    }
    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    for row in rows {
        let predicted = intercept
            + (0..x.cols())
                .map(|col| x.get2(*row, col).unwrap_or(0.0) * beta[col])
                .sum::<f64>();
        let err = y[*row] - predicted;
        let weight = weights.map(|values| values[*row]).unwrap_or(1.0);
        sum += weight * err * err;
        weight_sum += weight;
    }
    if weight_sum <= EPS {
        return f64::NAN;
    }
    sum / weight_sum
}
